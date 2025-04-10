from random import random
from copy import deepcopy
from typing import Dict, Optional, Union, Tuple, List

from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from tqdm import tqdm
from flwr.common.logger import log
from logging import WARNING
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common.typing import Parameters, FitIns, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from Common.model_utils import ndarrays_to_model, model_as_ndarrays, select_model
from Common.models import ResNet18
from Common.dataset_preparation_normal import CIFAR10PublicDataset

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class CustomFEDMD(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        inplace=True,
        server_loader=None,
        device=None,
        client_architectures: Dict[int, torch.nn.Module] = None,
    ):
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.server_loader = server_loader
        self.device = device
        self.client_architectures = client_architectures

        # Initialize server model
        self.server_model = ResNet18().to(self.device)
        self.server_model_params = parameters_to_ndarrays(initial_parameters)
        self.server_optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.001)

        # Checkpointing
        
        self.checkpoint_path = "server_checkpoints/checkpoint.pth"
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)



    def __repr__(self) -> str:
        """Compute a string representation of the strategy"""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep
    
    def num_fit_clients(self, num_available_clients):
        return super().num_fit_clients(num_available_clients)
    
    def num_evaluation_clients(self, num_available_clients):
        return super().num_evaluation_clients(num_available_clients)
    
    def initialize_parameters(self, client_manager):
        initial_parameters = self.initial_parameters

        return initial_parameters
    
    def evaluate(self, server_round, parameters):
        return super().evaluate(server_round, parameters)
    
    def configure_fit(self, server_round, parameters, client_manager):

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        config["server_logits"] = self.aggregated_server_logits
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results"""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        fit_config = self.on_fit_config_fn(server_round)

        aggregated_logits = self.aggregate_function_FEDMD(self.server_loader, fit_config, results)

        avg_log_parameters = ndarrays_to_parameters(aggregated_logits)

        metrics_aggregated = {}
        # Aggregate custom metrics if aggregation function was provided
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")     


        return avg_log_parameters, metrics_aggregated
    

    def aggregate_function_FEDMD(self, server_loader: DataLoader, fit_config: Dict[str, Scalar], results: List[Tuple[ClientProxy, FitRes]]):
        """Perform server-side knowledge distillation by aggregating logits."""

        Logits = []

        for _, fit_res in results:
            client_logits = []  # Store per-client logits

            # Load client model
            cid = fit_res.metrics['cid']
            client_model = select_model(self.client_architectures[int(cid)])
            client_model.to(self.device)
            ndarrays_to_model(client_model, parameters_to_ndarrays(fit_res.parameters))
            client_model.eval()

            # Get logits using the public dataset
            for idx, images, labels in CIFAR10PublicDataset():
                images = images.to(self.device)

                with torch.no_grad():
                    local_out = client_model(images)  # Shape: (batch_size, num_classes)

                client_logits.append(local_out.cpu())  # Store tensors directly

            # Convert client_logits to a single tensor before appending
            Logits.append(torch.cat(client_logits, dim=0))  # Shape: (num_samples, num_classes)

        # Aggregate the logits
        avg_logits = torch.mean(torch.stack(Logits), dim=0)  # Shape: (num_samples, num_classes)

        # Convert to a list of NumPy arrays (matches model_as_ndarrays)
        avg_logits_np_list = [logit.cpu().numpy() for logit in avg_logits]

        return avg_logits_np_list
    




    def aggregate_evaluate(self, server_round: int, results: list[tuple[ClientProxy, EvaluateRes]], 
                           failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]]
                           
                           ) -> tuple[Optional[float], dict[str, Scalar]]:

        """Aggregate evaluation losses using weighted average"""

        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Aggregated loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {}

        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated   
    



    # def aggregate_function_serverKD(self, server_loader: DataLoader, fit_config: Dict[str, Scalar], results: List[Tuple[ClientProxy, FitRes]]):
    #     """Perform server-side knowledge distillation."""
    #     # Load checkpoint if exists
    #     if os.path.exists(self.checkpoint_path):
    #         checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
    #         self.server_model.load_state_dict(checkpoint['model'])
    #         self.server_optimizer.load_state_dict(checkpoint['optimizer'])

    #     # Training setup
    #     epochs = fit_config.get("epochs", 1)
    #     alpha = fit_config.get("kd_alpha", 0.5)
    #     temp = fit_config.get("kd_temperature", 4.0)

    #     def kd_loss(student_output, labels, teacher_output):
    #         """Knowledge distillation loss."""
    #         return nn.KLDivLoss(reduction='batchmean')(
    #             F.log_softmax(student_output / temp, dim=1),
    #             F.softmax(teacher_output / temp, dim=1)
    #         ) * (alpha * temp**2) + F.cross_entropy(student_output, labels) * (1 - alpha)

    #     # Initialize server model with current parameters
    #     # ndarrays_to_model(self.server_model, self.server_model_params)
    #     # self.server_model.train()

    #     # Training loop
    #     for _ in range(epochs):
    #         for images, labels in server_loader:
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             # self.server_optimizer.zero_grad()
    #             # server_out = self.server_model(images)

    #             # Aggregate knowledge from clients
    #             loss = 0.0
    #             for _, fit_res in results:
    #                 cid = fit_res.metrics['cid']
    #                 # Initialize client-specific model
    #                 client_model = select_model(self.client_architectures[int(cid)])
    #                 client_model.to(self.device)
    #                 # Load client parameters
    #                 ndarrays_to_model(
    #                     client_model,
    #                     parameters_to_ndarrays(fit_res.parameters)
    #                 )
    #                 client_model.eval()
    #                 with torch.no_grad():
    #                     local_out = client_model(images)
    #                 # loss += kd_loss(server_out, labels, local_out)

    #             # Backpropagate and update server model
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), max_norm=5.0)
    #             self.server_optimizer.step()

    #     # Update server model parameters
    #     self.server_model_params = model_as_ndarrays(self.server_model)

    #     # Save checkpoint
    #     torch.save({
    #         'model': self.server_model.state_dict(),
    #         'optimizer': self.server_optimizer.state_dict()
    #     }, self.checkpoint_path)