from typing import Callable, Dict, List, OrderedDict
import os, sys
import flwr as fl
import torch
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Common.models import CNN_3layer_fc_model_removelogsoftmax, CNN_2layer_fc_model_removelogsoftmax, ResNet18
from flwr.common import Context
from Common.dataset import load_datasets
from Common.model_utils import train, train_with_kd, test, model_as_ndarrays, ndarrays_to_model, load_model

class FlowerClientProposedhomogeneous(fl.client.NumPyClient):
    def __init__(self, cid: int, trainloader: DataLoader, valloader:DataLoader, testloader:DataLoader, device: torch.device, 
                  model_architecture: str, algo_type) -> None:
        super().__init__()


        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.cid = cid
        self.model_architecture = model_architecture
        self._algo_type = algo_type

        # # Initialize models
        # if self.model_architecture == "CNN3L":
        #     self.client_model = CNN_3layer_fc_model_removelogsoftmax().to(self.device)
        # elif self.model_architecture == "CNN2L":
        #     self.client_model = CNN_2layer_fc_model_removelogsoftmax().to(self.device)
        
        # self.server_model = ResNet18().to(self.device)

        self.client_model, self.server_model = load_model(self.model_architecture, self.device, self._algo_type)

        print(self.server_model)

        
        # Initialize optimizer
        # self.optimizer = torch.optim.SGD(
        #     self.client_model.parameters(),
        #     lr=0.001,
        #     momentum=0.9,
        #     weight_decay=0.1
        # )
        self.optimizer = torch.optim.Adam(
            self.client_model.parameters(),
            lr=0.0001,
            weight_decay=0.1  # Keep weight decay for regularization
        )
        
        # Checkpoint setup

        # Checkpoint management
        self.checkpoint_dir = f"checkpoints/client_{cid}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        cid = self.cid
        return {'cid': int(cid)}

    
    def set_parameters(self, params):
        """Set the client model parameters from a list of NumPy ndarrays."""
        ndarrays_to_model(self.client_model, params)

    def get_parameters(self, config):
        """Return the client model parameters as a list of NumPy ndarrays."""
        return model_as_ndarrays(self.client_model)
    

    def fit(self, parameters, config):
        """Train with knowledge distillation and validate locally"""
        print(f"\n Client {self.cid}\n")
        # Set server model parameters
        ndarrays_to_model(self.server_model, parameters)
        
        # Extract config
        epochs = config["epochs"]
        current_round = config["server_round"]

        # Train with KD
        if current_round == 1:
            train_with_kd(
                self.client_model, self.server_model, config,
                self.trainloader, self.optimizer, epochs, self.device
            )
            self._save_checkpoint(current_round)
        else:
            self._load_checkpoint(current_round - 1)
            train_with_kd(
                self.client_model, self.server_model, config,
                self.trainloader, self.optimizer, epochs, self.device
            )
            self._save_checkpoint(current_round)

        # Local validation after training
        val_loss, val_acc = self._validate(config['client_kd_temperature'])
        print(f"Client {self.cid} validation - Loss: {val_loss:.4f}, Acc: {(val_acc*100):.2f}%")

        return self.get_parameters(config), len(self.trainloader), {'val_acc': val_acc, 'cid': float(self.cid)}
    
    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        current_round = config["server_round"]
        self._load_checkpoint(current_round)
        loss, acc = test(self.client_model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}
    
    def _validate(self, temp):
        """Run validation on local validation set"""
        return test(self.client_model, self.valloader, self.device, temp)

    def _test(self):
        """Run testing on global test set"""
        return test(self.client_model, self.testloader, self.device)
    



    def _save_checkpoint(self, round: int):
        torch.save(
            {
                "model": self.client_model.state_dict(),
                "optim": self.optimizer.state_dict()
            },
            os.path.join(self.checkpoint_dir, f"round_{round}.pth")
        )

    def _load_checkpoint(self, round: int):
        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, f"round_{round}.pth"),
            map_location=self.device
        )
        self.client_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])     
            
def gen_client_fn_homogeneous(
    trainloaders: list[DataLoader],
    valloaders: List[DataLoader],
    testloader: DataLoader,  # Global test set
    model_archs: dict[int, str],
    device: torch.device,
    algo_type
):
    def client_fn(cid: str) -> FlowerClientProposedhomogeneous:
        return FlowerClientProposedhomogeneous(
            cid=int(cid),
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloader,  # Same for all clients
            device=device,
            model_architecture=model_archs[int(cid)],
            algo_type=algo_type
        ).to_client()
    return client_fn





# def gen_client_fn(
#     trainloaders: list[DataLoader],
#     valloaders: list[DataLoader],  
#     testloader: DataLoader,  # Global test set
#     model_archs: dict[int, str],
#     device: torch.device
# ):
#     def client_fn(context: Context) -> FlowerClientProposed:
#         # Use `node_id` instead of `client_id`
#         cid = int(context.node_id)  

#         return FlowerClientProposed(
#             trainloader=trainloaders[cid],
#             valloader=valloaders[cid],
#             testloader=testloader,  
#             device=device,
#             cid=cid,
#             model_architecture=model_archs[cid]
#         ).to_client()

#     return client_fn


    
    

    
