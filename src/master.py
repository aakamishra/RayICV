import itertools

from ray.train import Trainer
from model_heap import ModelHeap
from metrics import MetricsCallback
from cross_validation import CrossValidationFoldGenerator
from worker import Worker


class Master:
    def __init__(self, dataset,
                 k_folds=5,
                 batch_size=32,
                 use_gpu=False,
                 backend="torch",
                 num_workers=3):
        """
        The Master class organizes the paritions of the
        dataset and also assigns each worker a configuration
        to train when it is finsihed with its task. It also keeps
        track of the termination condition.

        Params
        ------
        dataset: the data that we are training the master on
        k_folds: the number of folds that the user wants
        batch_size: the batch lenght for the train and val loader
        backend: specifies the ML framework to use for running the server
        num_workers: the number of workes to intialize overall

        Returns
        -------
        return: master object
        """
        self.num_workers = num_workers
        self.folds = k_folds
        self.use_gpu = use_gpu
        self.backend = backend
        self.dataset = dataset
        self.batch_size = batch_size

    def run(self, model_config):
        """
        The run function of the master class executes
        the logic with the initialized workers and returns
        the optimal configuration

        Params
        ------
        model_config: dictionary object containing information about
        the model and the parameters to tune over

        Returns
        -------
        return: a tuple of the best model configuration and the metric
        """

        # here we intialize the Ray trainer object
        self.trainer = Trainer(
            backend=self.backend,
            num_workers=self.num_workers,
            use_gpu=self.use_gpu)

        # create partition generator
        generator = CrossValidationFoldGenerator(self.dataset, self.folds,
                                                 self.batch_size, shuffle=True)

        # log initialization and create partitions
        print('Init Partition-Generator: ', generator)
        partitions = generator.generate()

        # get configuration parameters from the inputted model config
        keys, values = zip(*model_config['model_params'].items())

        # created a combinations list of all of the model configurations
        param_configs = [dict(zip(keys, v))
                         for v in itertools.product(*values)]

        # calculate the number of models we need to test and log
        num_models = len(param_configs)
        print('Parameter Configurations: ', param_configs)
        print('Number of Models: ', num_models)

        # initialize tracking data objects for metrics
        model_indices = [i for i in range(num_models)]
        model_train_losses = [0] * num_models
        model_val_losses = [0] * num_models

        # create model heap for suggesting scheduling of the training job
        heap = ModelHeap(model_indices)
        print("Heap Initialized: ", str(heap))

        # run execution loop of the objects
        while heap.folds_trained(self.folds) < num_models:
            # create config to pass worker
            model_indices = [
                heap.pop_model() if heap.heap[i][0] < self.folds else -1
                for i in range(self.num_workers)
            ]

            # create configuration dictionary and log
            config = {
                'data': partitions,
                'model_config': model_config,
                'model_indices': model_indices,
                'param_configs': param_configs,
            }
            print('Passing in config: ', config)

            # start thr trainer process using the initialized worker
            self.trainer.start()
            results = self.trainer.run(
                train_func=Worker.worker_func,
                config=config,
                callbacks=[MetricsCallback()]
            )

            # log the results given by each worker
            print(results)

            # after each process, calculate each metric for each worker
            for i in range(self.num_workers):
                agg_train_loss, agg_val_loss, _ = results[i]
                if agg_train_loss:
                    # calculate index with loss
                    model_index = model_indices[i][-1]
                    model_train_losses[model_index] += agg_train_loss
                    model_val_losses[model_index] += agg_val_loss

                    # keep track of folds given and trained per model conf.
                    current_folds = model_indices[i][0] + 1

                    # push back new model configuration update
                    heap.push_model(
                        folds=current_folds,
                        val_loss=model_val_losses[model_index] /
                        current_folds,
                        train_loss=model_train_losses[model_index] /
                        current_folds,
                        model=model_index)

            # log the current status of the heap
            print('Heap Update: ', str(heap))
            self.trainer.shutdown()
        print('val: ', model_val_losses, 'train: ', model_train_losses)
