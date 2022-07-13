from ray import train
from ray.train import Trainer, TrainingCallback
from typing import List, Dict

import torch
import torchmetrics

class PrintingCallback(TrainingCallback):
    def handle_result(self, results: List[Dict], **info):
        print(results)
        print(info)


def train_func(config):
    print(f"Worker {train.world_rank()}")
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    accuracy = torchmetrics.functional.accuracy(preds, target).item()
    train.report(accuracy=accuracy, config=config)

trainer = Trainer(backend="torch", num_workers=4)
trainer.start()
result = trainer.run(
    train_func,
    callbacks=[PrintingCallback()]
)
# [{'accuracy': 0.20000000298023224, '_timestamp': 1630716913, '_time_this_iter_s': 0.0039408206939697266, '_training_iteration': 1},
#  {'accuracy': 0.10000000149011612, '_timestamp': 1630716913, '_time_this_iter_s': 0.0030548572540283203, '_training_iteration': 1}]
trainer.shutdown()