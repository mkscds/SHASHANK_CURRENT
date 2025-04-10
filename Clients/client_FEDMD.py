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
from Common.dataset_preparation_normal import CIFAR10PublicDataset
from Common.model_utils import train, train_with_kd, test, model_as_ndarrays, ndarrays_to_model, load_model, train_FEDMD, train_FEDMD_KD

class FlowerClientFEDMD(fl.client.NumPyClient):
    def __init__(self, trainloader: DataLoader, valloader:DataLoader, testloader:DataLoader, device: torch.device, 
                 cid: int, model_architecture: str) -> None:
        super().__init__()


        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.cid = cid
        self.model_architecture = model_architecture
        self.public_dataloader = DataLoader(CIFAR10PublicDataset(), batch_size=1, shuffle=False)
        self.aggregated_logits = []  



        self.client_model, _ = load_model(self.model_architecture, self.device, self._algo_type)

        
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            self.client_model.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=0.1
        )
        self.publicoptimizer = torch.optim.Adam(self.client_model.parameters(), lr=0.001, momentum=0.9)
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
    
    def __initialize__local_model_public_dataset(self):
        # Transfer learning phase
        train_FEDMD(self.client_model, self.public_dataloader, self.publicoptimizer, 5, self.device) # public dataset
        train(self.client_model, self.trainloader, self.optimizer, 5, self.device) # private dataset

    def __getlogits_public_dataset(self):
        Local_logits = []
        for idx, images, labels in self.public_dataloader:
            images = images.to(self.device)  # Move to device

            with torch.no_grad():
                local_out = self.client_model(images)  # Shape: (batch_size, num_classes)

            # Convert tensor to NumPy and store in list
            Local_logits.extend(local_out.cpu().numpy())

        return Local_logits


    
    def fit(self, parameters, config):
        """Train with knowledge distillation and validate locally"""
        print(f"\n Client {self.cid}\n")


        # Receive aggregated logits from server.
        #self.aggregated_logits = []
        
        
        # Extract config
        epochs = config["epochs"]
        current_round = config["server_round"]

        # Train with KD

        # initialization of local model 

        if current_round == 1:
            self.__initialize__local_model_public_dataset() # initializing client model on public dataset
            self._save_checkpoint(current_round)
            # train(self.client_model, self.trainloader, self.optimizer, epochs, self.device) # training on private data
            logits = self.__getlogits_public_dataset() # send these logits to the server.

        else:
            self._load_checkpoint(current_round - 1)
            # KD with aggregated logits and public dataset --> Digest
            train_FEDMD_KD(self.client_model, self.aggregated_logits, config, self.public_dataloader, self.publicoptimizer, epochs, self.device)
            # normal training with private data --> Revisit
            train(self.client_model, self.trainloader, self.optimizer, 5, self.device)
            logits = self.__getlogits_public_dataset()



        # Local validation after training
        val_loss, val_acc = self._validate(config['kd_temperature'])
        print(f"Client {self.cid} validation - Loss: {val_loss:.4f}, Acc: {(val_acc*100):.2f}%")

        return logits, len(self.trainloader), {'val_acc': val_acc, 'cid': float(self.cid)}
    
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
    


def gen_client_fn_FEDMD(
    trainloaders: list[DataLoader],
    valloaders: List[DataLoader],
    testloader: DataLoader,  # Global test set
    model_archs: dict[int, str],
    device: torch.device
):
    def client_fn(cid: str) -> FlowerClientFEDMD:
        return FlowerClientFEDMD(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloader,  # Same for all clients
            device=device,
            cid=int(cid),
            model_architecture=model_archs[int(cid)]
        ).to_client()
    return client_fn