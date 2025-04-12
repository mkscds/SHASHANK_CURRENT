import os
from logging import WARNING
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    log,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from Common.model_utils import (
    model_as_ndarrays,
    ndarrays_to_model,
    load_model,
    select_model,
)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """ """


class CustomFedHomogen(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        client_architectures: Optional[List[str]] = None,
        algo_type: Optional[str] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.accept_failures = accept_failures
        self.server_loader = server_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_architectures = client_architectures or []
        self.algo_type = algo_type

        if not self.client_architectures:
            raise ValueError("Client architectures list cannot be empty")

        _, self.server_model = load_model(
            self.client_architectures[0], self.device, self.algo_type
        )
        self.server_model_params = parameters_to_ndarrays(initial_parameters)
        self.checkpoint_path = f"server_checkpoints/{self.algo_type}"
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def __repr__(self) -> str:
        return f"FedHomogen(accept_failures={self.accept_failures})"

    def aggregate_clients_logits(
        self, server_loader: DataLoader, results: List[Tuple[ClientProxy, FitRes]]
    ) -> torch.Tensor:
        all_client_logits = []
        weights = []
        total_examples = 0

        for _, fit_res in results:
            cid = fit_res.metrics["cid"]
            client_arch = self.client_architectures[int(cid)]

            client_model = select_model(client_arch)
            client_model.to(self.device)
            ndarrays_to_model(client_model, parameters_to_ndarrays(fit_res.parameters))

            client_logits = self.get_client_logits(server_loader, client_model)
            client_num_examples = fit_res.num_examples

            all_client_logits.append(client_logits)
            weights.append(client_num_examples)
            total_examples += client_num_examples

        if total_examples == 0:
            raise ValueError("No training examples available for aggregation")

        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        weights_tensor /= weights_tensor.sum()

        stacked_logits = torch.stack(all_client_logits, dim=0)
        return torch.sum(stacked_logits * weights_tensor.view(-1, 1, 1), dim=0)

    def get_client_logits(self, loader: DataLoader, model: nn.Module) -> torch.Tensor:
        model.eval()
        all_logits = []
        
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                logits = model(data)
                all_logits.append(logits)
        
        return torch.cat(all_logits, dim=0)

    def kd_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        temp: float,
        alpha: float,
    ) -> torch.Tensor:
        kl_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits.detach() / temp, dim=1)
        ) * (alpha * temp**2)

        ce_loss = F.cross_entropy(student_logits, labels) * (1 - alpha)
        return kl_loss + ce_loss

    def aggregate_function_serverKD(
        self,
        server_loader: DataLoader,
        server_round: int,
        fit_config: Dict[str, Scalar],
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Dict[str, float]:
        required_keys = ["server_kd_alpha", "server_kd_temperature", "server_learning_rate"]
        if not all(key in fit_config for key in required_keys):
            raise ValueError(f"Missing required keys in fit_config: {required_keys}")

        alpha = float(fit_config["server_kd_alpha"])
        temp = float(fit_config["server_kd_temperature"])
        epochs = int(fit_config.get("epochs", 5))

        # Initialize optimizer
        self.server_optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=float(fit_config["server_learning_rate"])
        )

        # Load checkpoint if available
        if server_round > 1:
            checkpoint_path = os.path.join(self.checkpoint_path, f"round_{server_round-1}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.server_model.load_state_dict(checkpoint["model"])
                self.server_optimizer.load_state_dict(checkpoint["server_optimizer"])

        self.server_model.to(self.device)
        self.server_model.train()

        # Get aggregated logits
        aggregate_logits = self.aggregate_clients_logits(server_loader, results)

        # Training loop
        total_batches = len(server_loader) * epochs
        avg_epoch_loss = 0.0

        with tqdm(total=total_batches, desc=f"Server KD Round {server_round}") as pbar:
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_idx, (images, labels) in enumerate(server_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    batch_size = images.size(0)

                    self.server_optimizer.zero_grad()
                    student_logits = self.server_model(images)

                    # Handle variable batch sizes
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    teacher_logits = aggregate_logits[start_idx:end_idx]

                    loss = self.kd_loss(student_logits, labels, teacher_logits, temp, alpha)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), max_norm=5.0)
                    self.server_optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({
                        "epoch": f"{epoch+1}/{epochs}",
                        "loss": f"{loss.item():.4f}"
                    })
                    pbar.update(1)

                avg_epoch_loss = epoch_loss / len(server_loader)
                tqdm.write(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

        # Update server parameters
        self.server_model_params = model_as_ndarrays(self.server_model)

        # Save checkpoint
        torch.save(
            {
                "model": self.server_model.state_dict(),
                "server_optimizer": self.server_optimizer.state_dict(),
            },
            os.path.join(self.checkpoint_path, f"round_{server_round}.pth")
        )

        return {"server_loss": avg_epoch_loss, "num_clients": len(results)}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}

        fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        self.aggregate_function_serverKD(self.server_loader, server_round, fit_config, results)

        # Convert parameters and metrics
        parameters = ndarrays_to_parameters(self.server_model_params)
        metrics = self._aggregate_metrics(results) if self.fit_metrics_aggregation_fn else {}

        return parameters, metrics

    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict[str, Scalar]:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        return self.fit_metrics_aggregation_fn(fit_metrics) if self.fit_metrics_aggregation_fn else {}