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
 		self.trainloader = trainloader
 		self.valloader = valloader
 		self.testloader = testloader
 		self.device = device
 		self.model_architecture = model_architecture
 		self.algo_type = algo_type
 		self.model, _ = load_model(self.model_architecture, self.device, self.algo_type)
 		self.optimizer = None


 	def get_parameters(self, config):
 		print(f" Client {self.cid} get_parameters")
 		return model_as_arrays(self.model)

 	def fit(self, parameters, config):
 		server_round = config["server_round"]
 		local_epochs = config["epochs"]
 		print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
 		ndarrays_to_model(self.model, parameters)

 		self.model, self.optimizer = train(self.model, self.trainloader, config, self.device)

 		val_loss, val_acc = self._validate(config['client_kd_temperature'])
 		print(f'Client {self.cid} validation Loss: {val_loss:.4f}, Validation Accuracy:{(val_acc*100):.2f}%')

 		return model_as_ndarrays(self.model) 


 	def evaluate(self, parameters, config):
 		print(f"[Client {self.cid}] evaluate, config: {config}")
 		ndarrays_to_model(self.model, parameters)
 		loss, accuracy = test(self.model, self.testloader, self.device, temp=1.0)
 		return float(loss), len(self.testloader), {'accuracy': float(accuracy)}


 	def _validate(self);

 		return test(self.model, self.valloader, self.device, temp=1.0)

 def gen_client_fn_fedavg(trainloaders: list[DataLoader],
 						  valloaders: list[DataLoader],
 						  testloader: DataLoader,
 						  model_archs: dict[int, str],
 						  device: torch.device,
 						  algo_type: str,):

 	def client_fn(cid: str) -> FlowerClientFedavg:
 		return FlowerClientFedavg(cid = int(cid),
 								  trainloader=trainloaders[int(cid)],
 								  valloader=valloaders[int(cid)],
 								  testloader=testloader,
 								  device=device,
 								  model_architecture=model_archs[int(cid)],
 								  algo_type=algo_type,).to_client()

