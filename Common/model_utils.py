from typing import List
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters
# from .models import CNN_3layer_fc_model_removelogsoftmax, CNN_2layer_fc_model_removelogsoftmax, ResNet18, Net

from .models import DeepNN_Hanu, DeepNN_Ram, DeepNN_Lax, ResNet18


def select_model(model_architecture):
    if model_architecture == "CNN_Hanu":
        client_model = DeepNN_Hanu(in_channels=3, num_classes=10)
    elif model_architecture == "CNN_Ram":
        client_model = DeepNN_Ram(in_channels=3, num_classes=10)
    elif model_architecture == "CNN_Lax":
        client_model = DeepNN_Lax(in_channels=3, num_classes=10)
    return client_model

def load_model(model_architecture, device, algo_type):
    model_mapping = {
        "CNN_Hanu": DeepNN_Hanu(in_channels=3, num_classes=10),
        "CNN_Ram": DeepNN_Ram(in_channels=3, num_classes=10),
        "CNN_Lax": DeepNN_Lax(in_channels=3, num_classes=10),
    }

    client_model = model_mapping.get(model_architecture, None)
    if not client_model:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")
    client_model = client_model.to(device)

    if algo_type == "homogeneous":
        server_model = client_model.__class__(in_channels=3, num_classes=10).to(device)
    elif algo_type == "heterogeneous":
        server_model = ResNet18().to(device)
    elif algo_type in  ["fedavg", "fedmd"]:
        server_model = None
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")
    
    return client_model, server_model

def create_model_architectures(available_architectures, algo_type, num_clients):
    if algo_type in ["homogeneous", "fedavg"]:
        selected_model = random.choice(available_architectures)  # Select one model randomly
        model_architectures = {i: selected_model for i in range(num_clients)}
    elif algo_type in  ["heterogeneous", "fedmd"]:
        model_architectures = {i: random.choices(available_architectures, weights=[0.3, 0.6, 0.1])[0] for i in range(num_clients)}

    return model_architectures

# def create_model_architectures(available_architectures, algo_type, num_clients):
#     if algo_type == "homogeneous":
#         selected_model = random.choice(available_architectures)  # Select one model randomly
#         model_architectures = {i: selected_model for i in range(num_clients)}
#     elif algo_type in  ["heterogeneous", "FEDMD"]:
#         model_architectures = {i: random.choices(available_architectures, weights=[0.3, 0.6, 0.1])[0] for i in range(num_clients)}
#     elif algo_type == "fedavg":
#         selected_model = available_architectures[2]
#         model_architectures = {i: selected_model for i in range(num_clients)}


    

def model_as_ndarrays(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def ndarrays_to_model(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays"""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# borrowed from PyTorch quickstart example
def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()

def train_FEDMD(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for _, images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()

def train_FEDMD_KD(student, aggregated_logits, kd_config, trainloader, optim, epochs, device: str):
    """Train network on the training set using KD."""
    alpha = kd_config["kd_alpha"]
    temp = kd_config["kd_temperature"]

    def kd_loss(student_output, labels, teacher_output):
        # KD loss
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/temp, dim=1), 
                                                  F.softmax(teacher_output/temp, dim=1)) * (alpha * temp**2) + \
               F.cross_entropy(student_output, labels) * (1 - alpha)

    print(f'\n STARTING CLIENT TRAINING')

    student.train()
    for epoch in range(epochs):  # Track epoch number
        # running_loss = 0.0
        # running_correct = 0
        # total_samples = 0

        for _, images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()

            # Forward pass through student and teacher
            s_out = student(images)
            # Compute loss and backpropagate
            loss = kd_loss(s_out, labels, aggregated_logits)
            loss.backward()
            optim.step()


# def load_model(model_architecture, device, algo_type):
#     model_mapping = {
#         "CNN3L": CNN_3layer_fc_model_removelogsoftmax(),
#         "CNN2L": CNN_2layer_fc_model_removelogsoftmax(),
#         "Net": Net(in_channels=3, num_classes=10),
#     }

#     client_model = model_mapping.get(model_architecture, None)
#     if not client_model:
#         raise ValueError(f"Unsupported model architecture: {model_architecture}")
#     client_model = client_model.to(device)

#     if algo_type == "homogeneous":
#         server_model = client_model.__class__(in_channels=3, num_classes=10).to(device)
#         if model_architecture == "Net":
#             server_model = client_model.__class__(in_channels=3, num_classes=10).to(device)
#         else:
#             server_model = client_model.__class__().to(device)

#     elif algo_type == "heterogeneous":
#         server_model = ResNet18().to(device)
#     elif algo_type == "fedavg":
#         server_model = None
#     else:
#         raise ValueError(f"Unsupported algorithm type: {algo_type}")
#     return client_model, server_model

    #         # Accumulate metrics
    #         batch_size = images.size(0)
    #         running_loss += loss.item() * batch_size
    #         _, preds = torch.max(s_out, 1)  # Get predictions
    #         running_correct += (preds == labels).sum().item()
    #         total_samples += batch_size

    #     # Calculate epoch metrics
    #     epoch_loss = running_loss / total_samples
    #     epoch_acc = running_correct / total_samples

    #     print(f'Epoch [{epoch+1}/{epochs}]: '
    #           f'Train Loss: {epoch_loss:.4f}, '
    #           f'Train Accuracy: {epoch_acc*100:.2f}%')

    # print('CLIENT TRAINING COMPLETE\n')

# def train_with_kd(student, teacher, kd_config, trainloader, optim, epochs, device:str):
#     """Train network on the training set using KD."""
#     alpha = kd_config["kd_alpha"]
#     temp = kd_config["kd_temperature"]
#     def kd_loss(student_output, labels, teacher_output):
#         # KD loss 
#         return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/temp, dim = 1), F.log_softmax(teacher_output/temp, dim=1))*(alpha*temp**2) + F.cross_entropy(student_output, labels)*(1 - alpha)

#     print(f'\n STARTING CLIENT TRAINING')

#     student.train()
#     teacher.eval()
#     for _ in range(epochs):
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)
#             optim.zero_grad()
#             s_out = student(images)

#             # pass the same batch though teacher model
#             with torch.inference_mode():
#                 t_out = teacher(images)

#             loss = kd_loss(s_out, labels, t_out)
#             loss.backward()
#             optim.step()


def train_with_kd(student, teacher, kd_config, trainloader, optim, epochs, device: str):
    """Train network on the training set using KD."""
    alpha = kd_config["client_kd_alpha"]
    temp = kd_config["client_kd_temperature"]

    def kd_loss(student_output, labels, teacher_output):
        # KD loss
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/temp, dim=1), 
                                                  F.softmax(teacher_output/temp, dim=1)) * (alpha * temp**2) + \
               F.cross_entropy(student_output, labels) * (1 - alpha)

    print(f'\n STARTING CLIENT TRAINING')

    student.train()
    teacher.eval()

    for epoch in range(epochs):  # Track epoch number
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()

            # Forward pass through student and teacher
            s_out = student(images)
            with torch.inference_mode():
                t_out = teacher(images)

            # Compute loss and backpropagate
            loss = kd_loss(s_out, labels, t_out)
            loss.backward()
            optim.step()

            # Accumulate metrics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(s_out, 1)  # Get predictions
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples

        print(f'Epoch [{epoch+1}/{epochs}]: '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Train Accuracy: {epoch_acc*100:.2f}%')

    print('CLIENT TRAINING COMPLETE\n')




def train_one_client(student, teacher, dataloader, config, server_optimizer, device):
    alpha = config["server_kd_alpha"]
    temp = config["server_kd_temperature"]

    def kd_loss(student_logits, labels, teacher_logits):
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits /temp, dim =1),
                                                   F.softmax(teacher_logits / temp, dim =1))*(alpha* temp**2) + \
                                                   F.cross_entropy(student_logits, labels)*(1 - alpha)
    

    student.train()
    teacher.eval()
    loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        server_optimizer.zero_grad()
        s_out = student(images)
        with torch.inference_mode():
            t_out = teacher(images)

        loss = kd_loss(s_out, labels, t_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        server_optimizer.step()

def test(net, testloader, device: str, temp=1.0):
    """Validate the network on the entire test set with temperature scaling"""
    net.to(device).eval()
    correct, total, loss = 0, 0, 0.0
    
    with torch.inference_mode():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            # Forward pass with temperature scaling
            logits = net(images) / temp  # Critical change
            
            # Calculate loss (now aligned with KD training)
            loss += F.cross_entropy(logits, labels).item()
            
            # Get predictions (temperature doesn't affect argmax)
            _, predicted = torch.max(logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average loss across all batches
    avg_loss = loss / len(testloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def get_server_initial_parameters(net, server_dataloader, server_epochs, device: str):

                                   # TODO: Confirm on criterion for local_outputs and server_model_outputs
    optimizer = torch.optim.SGD(net.parameters(),lr=0.001)
    print('\n INITIALIZING SERVER MODEL \n')
    train(net.to(device), server_dataloader, optimizer, server_epochs, device)

    print('\n COMPLETED INITIALIZING SERVER MODEL \n')
    numpy_parameters = model_as_ndarrays(net)
    # server_initial_parameters = fl.common.ndarrays_to_parameters(numpy_parameters)

    return numpy_parameters





