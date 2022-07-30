from ray.train import Trainer
from ray.train.examples.train_fashion_mnist_example import train_func

from metrics import MetricsCallback
from cross_validation import CrossValidationFoldGenerator
from worker import Worker

class Master:
    def __init__(self, dataset, k_folds=5, batch_size=32, use_gpu=False,
                 backend="torch"):
        self.num_workers = k_folds
        self.folds = k_folds
        self.use_gpu = use_gpu
        self.backend = backend
        self.dataset = dataset
        self.batch_size = batch_size

    def run(self, model_config):
        self.trainer = Trainer(backend=self.backend,
            num_workers=self.num_workers, use_gpu=self.use_gpu)
        # create partition generator
        generator = CrossValidationFoldGenerator(self.dataset, self.folds,
                                                 self.batch_size, shuffle=True)
        print(f"Init Partition-Generator: {generator}")
        partitions = generator.generate()

        # create config to pass worker
        config = {"data" : partitions, "model_config" : model_config}
        print(f"Passing in config: {config}")


        self.trainer.start()
        results = self.trainer.run(train_func=Worker.worker_func, config=config,
                                   callbacks=[MetricsCallback()])
        print(results)
        self.trainer.shutdown()
