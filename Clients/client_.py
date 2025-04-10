from typing import Callable, Dict, List, OrderedDict

import os, sys
import flwr as fl
from flwr.common import Context
import torch
from torch.utils.data import DataLoader
from flwr.common import Scalar

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Common.models import CNN_2layer_fc_model_removelogsoftmax, CNN_3layer_fc_model_removelogsoftmax
from Common.dataset import load_datasets
from Common.model_utils import train, train_with_kd, test, model_as_ndarrays, ndarrays_to_model, load_model

class FLowerClientProoposedHomogeneous(fl.client.NumPyClient):
    def __init__(self, trainloader: DataLoader, valloader: DataLoader, testloader: DataLoader, device: torch.device, 
                 cid: int, model_architecture: str, algo_type: str) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.cid = cid
        self.model_architecture = model_architecture
        self.algo_type = algo_type

        self.client_model, self.server_model = load_model(self.model_architecture, self.device, self.algo_type)

        self.optimizer = torch.optim.Adam(self.client_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)

        self.checkpoint_dir = f"checkpoints/client_{cid}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def get_properties(self, config: Dict[str: Scalar]) -> Dict[str, Scalar]:
        cid = self.cid
        return {'cid': int(cid)}
    
    def set_parameters(self, params):
        """Set the client parameters from a list of NumPy ndarrays"""
        ndarrays_to_model(self.client_model, params)

    def get_parameters(self, config):
        """Return the client model parameters as a list of NumPy ndarrays"""
        return model_as_ndarrays(self.client_model)

    def fit(self, parameters, config):
        """Train with KD and validate locally"""
        print(f'\n Client {self.cid} \n')
        ndarrays_to_model(self.server_model, parameters)

        # config
        epochs = config["epochs"]
        current_round = config["server_round"]

        if current_round == 1:
            # training with random initialization
            train_with_kd(self.client_model, self.server_model, config, self.trainloader, self.optimizer, epochs, self.device)
            self._save_checkpoint(current_round)

        else:
            self._load_checkpoint(current_round - 1)
            train_with_kd(self.client_model, self.server_model, config, self.trainloader, self.optimizer, epochs, self.device)
            self._save_checkpoint(current_round)

        # Local validation after training
        val_loss, val_acc = self._validate(config['kd_temperature'])
        print(f'Client {self.cid} validation Loss: {val_loss:.4f}, Validation Accuracy: {(val_acc*100):.2f}')

        return self.get_parameters(config), len(self.trainloader), {'val_acc': val_acc, 'cid': float(self.cid)}
    

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters"""
        current_round = config["server_round"]
        self._load_checkpoint(current_round)
        loss, acc = test(self.client_model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}
    

    def _validate(self, temp):
        """Run validation on local validation set"""
        return test(self.client_model, self.valloader, self.device, temp)
    

    def _save_checkpoints(self, round: int):
        torch.save(
            {
                "model": self.client_model.state_dict(),
                "optim": self.optimizer.state_dict()
            }, os.path.join(self.checkpoint_dir, f"round_{round}.pth"))

    def _load_checkpoint(self, round: int):
        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, f"round_{round}.pth"),
            map_location=self.device)
        self.client_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])


def gen_client_fn_homogenous(
        trainloaders: List[DataLoader],
        valloaders: List[DataLoader],
        testloader: DataLoader,
        model_archs: dict[int, str],
        device: torch.device,
        algo_type:str):
    
    def client_fn(cid: str) -> FLowerClientProoposedHomogeneous:
        return FLowerClientProoposedHomogeneous(
             trainloader=trainloaders[int(cid)],
             valloader=valloaders[int(cid)],
             testloader=testloader,
             device=device,
             model_architecture=model_archs[int(cid)],
             algo_type = algo_type).to_client()
    
     
    return client_fn

