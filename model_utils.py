import random
import numpy as np
import flwr as fl
from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging

from .models import DeepNN_Hanu, DeepNN_Lax, DeepNN_Ram, ResNet18

# Setup logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def selected_model(model_architecture):
    if model_architecture == "CNN_Hanu":
        client_model = DeepNN_Hanu(in_channels=3, num_classes=10)
    elif model_architecture == "CNN_Ram":
        client_model = DeepNN_Ram(in_channels=3, num_classes=10)
    elif model_architecture == "CNN_Lax":
        client_model = DeepNN_Lax(in_channels=3, num_classes=10)
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")
    return client_model

def load_model(model_architecture, device, algo_type):
    model_mapping = {
        "CNN_Hanu": DeepNN_Hanu,
        "CNN_Ram": DeepNN_Ram,
        "CNN_Lax": DeepNN_Lax,
    }

    # Get model class
    client_model_class = model_mapping.get(model_architecture)
    if client_model_class is None:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    # Instantiate client model
    client_model = client_model_class(in_channels=3, num_classes=10).to(device)

    # Initialize server model based on algorithm type
    if algo_type == "homogeneous":
        server_model = client_model_class(in_channels=3, num_classes=10).to(device)
    elif algo_type == "heterogeneous":
        server_model = ResNet18().to(device)
    elif algo_type in ["fedavg", "fedmd"]:
        server_model = None
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")

    return client_model, server_model


def create_model_architectures(available_architectures, algo_type, num_clients):
    if algo_type in ["homogeneous", "fedavg"]:
        # All clients use the same architecture
        chosen_model = random.choice(available_architectures)
        model_architectures = {i: chosen_model for i in range(num_clients)}
    elif algo_type in ["heterogeneous", "fedmd"]:
        # Assign architectures with equal probability
        num_models = len(available_architectures)
        weights = [1/num_models] * num_models  # Uniform distribution
        model_architectures = {
            i: random.choices(available_architectures, weights=weights)[0]
            for i in range(num_clients)
        }
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")

    return model_architectures

def model_as_ndarrays(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def ndarrays_to_model(model, params):
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def train(model, trainloader, config, device):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 10),
        gamma=config.get('lr_gamma', 0.1)
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    logging.info("Training started.")
    for epoch in range(config['epochs']):
        running_loss = 0.0
        total_samples = 0  # Track total samples for averaging
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size  # Weight by batch size
            total_samples += batch_size
            
            progress_bar.set_postfix(
                loss=loss.item(), 
                lr=optimizer.param_groups[0]['lr']
            )

        avg_loss = running_loss / total_samples  # Correct averaging
        logging.info(
            f"Epoch {epoch+1}/{config['epochs']} - "
            f"Avg Loss: {avg_loss:.4f} - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()

    logging.info("Training completed.")
    return model, optimizer

def train_with_kd(client_model, server_model, config, trainloader, optimizer, device):
    alpha = config['client_kd_alpha']
    temp = config['client_kd_temperature']
    epochs = config['epochs']

    def kd_loss(student_logits, student_labels, teacher_logits):
        # KLDivLoss expects log probabilities for the student
        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1)
        )
        # Cross-entropy for ground truth labels
        ce_loss = F.cross_entropy(student_logits, student_labels)
        return kl_loss * (alpha * temp**2) + ce_loss * (1 - alpha)

    logging.info('Starting client training with KD')
    client_model.train()
    server_model.eval()

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        loop = tqdm(trainloader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            optimizer.zero_grad()
            student_logits = client_model(images)
            
            with torch.no_grad():
                teacher_logits = server_model(images)

            loss = kd_loss(student_logits, labels, teacher_logits)
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * batch_size
            _, preds = torch.max(student_logits, 1)
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        logging.info(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Loss: {epoch_loss:.4f} - "
            f"Accuracy: {epoch_acc * 100:.2f}%"
        )

    logging.info('Client training with KD completed\n')
    
    return client_model



