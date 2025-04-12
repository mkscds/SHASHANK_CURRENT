from typing import Callable, Dict, List, OrderedDict
import os, sys
import flwr as fl

import torch
from flwr.common import Scalar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Common.model_utils import train, train_with_kd, test, model_as_ndarrays, ndarrays_to_model

from torch.utils.data import DataLoader
from Common.model_utils import load_model
  

class FlowerClientHomogen(fl.client.NumPyClient):
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

		self.checkpoint_dir = f'checkpoints/client_{cid}'
		os.makedirs(self.checkpoint_dir, exist_ok=True)


	def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
		cid = self.cid
		return {'cid': int(cid)}


	def set_parameters(self, params):
		ndarrays_to_model(self.client_model, params)

	def get_parameters(self, config):
		return model_as_ndarrays(self.client_model)


	def fit(self, parameters, config):

		print(f"\n Client {self.cid} \n")

		ndarrays_to_model(self.server_model, parameters)

		epochs = config["epochs"]
		current_round = config["server_round"]

		self.optimizer = torch.optim.SGD(client_model.parameters(), lr=config['learning'])

		if current_round == 1:
			train_with_kd(self.client_model, self.server_model, config, self.trainloader, self.optimizer, self.device)
			self._save_checkpoint(current_round)

		else:
			self._load_checkpoint(current_round - 1)
			train_with_kd(self.client_model, self.server_model, self.device)
			self._save_checkpoint(current_round)


		val_loss, val_acc = self._validate(config['server_kd_temperature'])
		print(f"Client {self.cid} Validation Loss: {val_loss:.4f}, Validation Accuracy: {(val_acc*100):.2f}")

		return self.get_parameters(config), len(self.trainloader), {'val_acc': val_acc, 'cid': float(self.cid)}



	def evaluate(self, parameters, config: Dict[str, Scalar]):

		current_round = config['server_round']
		self._load_checkpoint(current_round)
		loss, acc = test(self.client_model, self.testloader, self.device)
		return float(loss), len(self.valloader.dataset), {'accuracy': float(acc)}


	def _validate(self, temp):

		return test(self.client_model, self.valloader, self.device)


	def _save_checkpoint(self, round: int):
		torch.save(
			{
				"model": self.client_model.state_dict(),
				"optim": self.optimizer.state_dict()
			},
			os.path.join(self.checkpoint_dir, f"round_{round}.pth"))


	def _load_checkpoints(self, round: int):
		checkpoint = torch.load(
			os.path.join(self.checkpoint_dir, f"round_{round}.pth"),
			map_location="cpu")

		self.client_model.load_state_dict(checkpoint['model'])
		self.optimizer.load_state_dict(checkpoint["optim"])
		self.client_model = self.client_model.to(self.device)


def gen_client_fn_homogen(trainloaders: List[DataLoader],
							valloaders: List[DataLoader],
							testloader: DataLoader,
							model_archs: dict[int, str],
							device: torch.device,
							algo_type: str):

	def client_fn(cid: str) -> FlowerClientHomogen:
		return FlowerClientHomogen(trainloader=trainloaders[int(cid)],
							   valloader = trainloaders[int(cid)],
							   testloader = DataLoader,
							   device = device,
							   cid = int(cid),
							   model_architecture = model_archs[int(cid)],
							   algo_type=algo_type).to_client()


	return client_fn







