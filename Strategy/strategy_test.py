from logging import WARNING
from typing import Callable, Optional, Union, Tuple, Dict, List

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.utils.data import DataLoader

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy
from flwr.server.strategy.fedavg import FedAvg

from Common.model_utils import (
    train,
    train_with_kd,
    test,
    model_as_ndarrays,
    ndarrays_to_model,
    load_model,
    select_model,
)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """ Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients connected to the server. `min_evaluate_clients`
must be set to a value larger than or equal to the values of `min_fit_clients` and `min_evaluate_clients`."""


class CustomFedhetHomogenous(FedAvg):
    def __init__(self, *, fraction_fit=1, fraction_evaluate=1, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
            evaluate_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None, accept_failures=True, initial_parameters=None,
            fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None, inplace=True, server_loader=None, device=None,
            client_architectures: Dict[int, torch.nn.Module] = None, algo_type:str =None):
        super().__init__()

        if (min_fit_clients > min_available_clients or min_evaluate_clients > min_available_clients):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
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
        self._algo_type = algo_type

        _, self.server_model = load_model(self.client_architectures[0], self.device, self._algo_type)
        self.server_model_params = parameters_to_ndarrays(initial_parameters)
        self.server_optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.001)

        self.checkpoint_path = f"server_checkpoints/{self._algo_type}"
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def __repr__(self) -> str:
        rep = f"FedHet(accept_failures={self.accept_failures})"
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
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return super().configure_evaluate(server_round, parameters, client_manager)
        
    def _get_clients_logits_(self, server_loader, results, temperature=1.0):
        all_client_logits = []
        
        # Collect logits from all clients
        for _, fit_res in results:
            cid = fit_res.metrics['cid']
            
            # 1. Select and prepare client model
            client_model = select_model(self.client_architectures[int(cid)])
            client_model.to(self.device)
            ndarrays_to_model(client_model, parameters_to_ndarrays(fit_res.parameters))
            
            # 2. Get logits for this client (shape: [num_samples, num_classes])
            client_logits = self.get_client_logits(server_loader, client_model, self.device)
            all_client_logits.append(client_logits)
        
        # 3. Aggregate logits across clients
        stacked_logits = torch.stack(all_client_logits, dim=0)  # [num_clients, num_samples, num_classes]
        aggregated_logits = torch.mean(stacked_logits, dim=0)
        
        return aggregated_logits

    def aggregate_function_serverKD(self, server_loader: DataLoader, server_round: int, fit_config: Dict[str, Scalar], results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[str, Scalar]:
        """Perform server-side knowledge distillation with progress tracking"""

        # Extract hyperparameters
        alpha = float(fit_config["server_kd_alpha"])
        temp = float(fit_config["server_kd_temperature"])
        epochs = int(fit_config.get("epochs", 5))
        device = self.device

        # Define KD loss (modified to show temperature scaling clearly)
        def kd_loss(student_logits: torch.Tensor, labels: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1)
            ) * (alpha * temp**2)
            
            ce_loss = F.cross_entropy(student_logits, labels) * (1 - alpha)
            return kl_loss + ce_loss

        # Load previous checkpoint if available
        if server_round > 1 and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(
                os.path.join(self.checkpoint_path, f"round_{server_round - 1}.pth"),
                map_location="cpu"
            )
            self.server_model.load_state_dict(checkpoint["model"])
            self.server_optimizer.load_state_dict(checkpoint["optimizer"])
            self.server_model = self.server_model.to(device)

        # Prepare model and optimizer
        ndarrays_to_model(self.server_model, self.server_model_params)
        self.server_model.train()
        
        # Get aggregated logits from clients
        aggregated_logits = self._get_clients_logits_(server_loader=server_loader, results=results, temperature=temp).to(device)

        # Training loop with TQDM
        total_batches = len(server_loader) * epochs
        with tqdm(total=total_batches, desc=f"Server KD Round {server_round}") as pbar:
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_idx, (images, labels) in enumerate(server_loader):
                    images, labels = images.to(device), labels.to(device)
                    
                    # Forward pass
                    self.server_optimizer.zero_grad()
                    student_logits = self.server_model(images)
                    
                    # Compute loss
                    batch_teacher_logits = aggregated_logits[batch_idx * images.shape[0] : (batch_idx + 1) * images.shape[0]]
                    loss = kd_loss(student_logits, labels, batch_teacher_logits)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), max_norm=5.0)
                    self.server_optimizer.step()
                    
                    # Update progress
                    epoch_loss += loss.item()
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{epochs}',
                        'loss': f'{loss.item():.4f}'
                    })
                    pbar.update(1)
                
                # Epoch summary
                avg_epoch_loss = epoch_loss / len(server_loader)
                tqdm.write(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

        self.server_model_params = model_as_ndarrays(self.server_model)

        torch.save({
            'model': self.server_model.state_dict(),
            'optimizer': self.server_optimizer.state_dict()}, os.path.join(self.checkpoint_path, f"round_{server_round}.pth"))

        # Return training metrics
        return {"server_loss": avg_epoch_loss, "num_clients": len(results), "temperature": temp, "alpha": alpha}

    def aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        fit_config = self.on_fit_config_fn(server_round)

        self.aggregate_function_serverKD(self.server_loader, server_round, fit_config, results)

        KD_parameters = ndarrays_to_parameters(self.server_model_params)

        metrics_aggregated = {}

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return KD_parameters, metrics_aggregated


	