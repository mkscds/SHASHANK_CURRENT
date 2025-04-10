from typing import Callable, Dict, List, OrderedDict
import os, sys
import flwr as fl
import torch
from flwr.common import Scalar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Common.model_utils import train, train_with_kd, test, model_as_ndarrays, ndarrays_to_model
from torch.utils.data import DataLoader
from Common.model_utils import load_model

class FlowerClientFedavg(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, valloader, testloader, device, model_architecture, algo_type):
        self.cid = cid
        # self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.model_architecture = model_architecture
        self._algo_type = algo_type
        self.net, _ = load_model(self.model_architecture, self.device, self._algo_type)
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.1
        )



    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return model_as_ndarrays(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["epochs"]
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        ndarrays_to_model(self.net, parameters)
        train(self.net, self.trainloader, self.optimizer, local_epochs, self.device)
        # train(self.net, self.trainloader,local_epochs,server_round,)
        val_loss, val_acc = self._validate(config['client_kd_temperature'])
        print(f"Client {self.cid} validation - Loss: {val_loss:.4f}, Acc: {(val_acc*100):.2f}%")
        return model_as_ndarrays(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        ndarrays_to_model(self.net, parameters)       
        loss, accuracy = test(self.net, self.testloader, self.device, temp=1.0)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
    
    def _validate(self):
        """Run validation on local validation set"""
        return test(self.net, self.valloader, self.device, temp=1.0)    

def gen_client_fn_fedavg(
    trainloaders: list[DataLoader],
    valloaders: List[DataLoader],
    testloader: DataLoader,  # Global test set
    model_archs: dict[int, str],
    device: torch.device,
    algo_type,
):
    def client_fn(cid: str) -> FlowerClientFedavg:
        return FlowerClientFedavg(
            cid=int(cid),
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloader,  # Same for all clients
            device=device,
            model_architecture=model_archs[int(cid)],
            algo_type=algo_type,
        ).to_client()
    return client_fn

