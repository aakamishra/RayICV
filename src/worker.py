from ray import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from workflow import Workflow



class Worker:
	def __init__(self, data, model_config):
		# initialize device worker
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		print(f'Init Worker: {train.world_rank()}, Using device: {self.device}')

		self.train_loader, self.val_loader = data

		# import config params
		self.model = train.torch.prepare_model(model_config["model"]()).to(self.device)
		self.optimizer = model_config["optimizer"](self.model.parameters())
		self.epochs = model_config["epochs"]


	def worker_run(self):

		# epoch runner
		for epoch in range(self.epochs):
			train_loss = Workflow.train(self.model, self.device, self.train_loader, self.optimizer)
			val_loss, val_accuracy = Workflow.val(self.model, self.device, self.val_loader)
			train.report(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy, worker_id=train.world_rank())

	@staticmethod
	def worker_func(config):
		Worker(config["data"][train.world_rank()], config["model_config"]).worker_run()
		





