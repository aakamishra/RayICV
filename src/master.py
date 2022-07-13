from ray.train import Trainer
from ray.train.examples.train_fashion_mnist_example import train_func

from metrics import MetricsCallback


class Master:
    def __init__(self, num_workers, use_gpu, backend):
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.backend = backend

    def run(self, config):
        self.trainer = Trainer(backend=self.backend, num_workers=self.num_workers, use_gpu=self.use_gpu)
        self.trainer.start()
        results = self.trainer.run(
            train_func=train_func,
            config=config,
            callbacks=[MetricsCallback()],
        )
        print(results)
        self.trainer.shutdown()
