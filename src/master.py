import itertools

from ray.train import Trainer
from ray.train.examples.train_fashion_mnist_example import train_func
from model_heap import ModelHeap
from metrics import MetricsCallback
from cross_validation import CrossValidationFoldGenerator
from worker import Worker

class Master:
    def __init__(self, dataset, k_folds=5, batch_size=32, use_gpu=False,
                 backend="torch", num_workers=3):
        self.num_workers = 3
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

        keys, values = zip(*model_config["model_params"].items())
        param_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        num_models = len(param_configs)
        print("Parameter Configurations: ", param_configs)
        print("Number of Models: ", num_models)
        model_indices = [i for i in range(num_models)]
        model_train_losses = [0] * num_models
        model_val_losses = [0] * num_models

        heap = ModelHeap(model_indices)
        print("Heap Initialized: ", str(heap))
        while heap.folds_trained(self.folds) < num_models:
            # create config to pass worker
            model_indices = [heap.pop_model() if heap.heap[i][0] < self.folds else -1 for i in range(self.num_workers)]
            config = {"data" : partitions, "model_config" : model_config, "model_indices" : model_indices, "param_configs" : param_configs}
            print(f"Passing in config: {config}")
            self.trainer.start()
            results = self.trainer.run(train_func=Worker.worker_func, config=config,
                                       callbacks=[MetricsCallback()])
            print(results)
            for i in range(self.num_workers):
                agg_train_loss, agg_val_loss, avg_val_acc = results[i]
                if agg_train_loss is not None:

                    model_index = model_indices[i][-1]
                    model_train_losses[model_index] += agg_train_loss
                    model_val_losses[model_index] += agg_val_loss

                    current_folds = model_indices[i][0] + 1
                    heap.push_model(folds=current_folds, 
                                    val_loss=model_val_losses[model_index]/current_folds, 
                                    train_loss=model_train_losses[model_index]/current_folds, 
                                    model=model_index)
            print("Heap Update: ", str(heap))
            self.trainer.shutdown()
        print("val: ", model_val_losses, "train: ", model_train_losses)
