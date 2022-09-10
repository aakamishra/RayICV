import numpy as np
from ray import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from workflow import Workflow


class Worker:
    def __init__(self, data, model_config, param_config):
        # initialize device worker
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        print(f'Init Worker: {train.world_rank()}, Using device: {self.device}')

        self.train_loader, self.val_loader = data

        # import config params
        print('Intializing Model with Config: ', param_config)
        self.model = train.torch.prepare_model(
            model_config['model'](param_config)
        ).to(self.device)
        self.optimizer = model_config['optimizer'](self.model.parameters())
        self.epochs = model_config['epochs']

    def worker_run(self):
        # epoch runner
        agg_train_loss, agg_val_loss, avg_val_acc = 0, 0, 0
        val_acc_arr = []
        for epoch in range(self.epochs):
            train_loss = Workflow.train(self.model, self.device,
                                        self.train_loader, self.optimizer)
            val_loss, val_accuracy = Workflow.val(self.model, self.device,
                                                  self.val_loader)
            train.report(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                         val_accuracy=val_accuracy,
                         worker_id=train.world_rank())
            agg_train_loss += train_loss
            agg_val_loss += val_loss
            val_acc_arr.append(val_accuracy)

        avg_val_acc = np.mean(val_acc_arr)
        return agg_train_loss, agg_val_loss, avg_val_acc

    @staticmethod
    def worker_func(config):
        working_model = config['model_indices'][train.world_rank()]
        print('Working_model: ', working_model)
        model_index = working_model[-1]
        if model_index < 0:
            return None, None, None
        fold_index = working_model[0]
        print(f'Creating new worker assignment with model assignment \
            {model_index} and fold {fold_index}')
        worker = Worker(
            config['data'][fold_index],
            config['model_config'],
            config['param_configs'][model_index]
        )
        return worker.worker_run()
