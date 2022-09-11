from master import Master


class RayCrossValidation:
    def __init__(
        self,
        model,
        dataset,
        parameters,
        folds,
        optimizer,
        epochs=10,
        batch_size=32,
        use_gpu=False,
        backend="torch",
        num_workers=3,
    ):
        """
        The software shim for the RayCrossValidation API takes in the
        necessary parameters to run the cross validation algorithm
        and allows the user to specify the basic parameters for
        the distributed environment they want the shim to run on.

        Params
        ------
        model: PyTorch Model that the user want to train
            in-terms of the class object
        dataset: data that is to be shared amongst the workers and split
            into folds during cross validation
        parameters: the parameter space that is need for the
            model as input so that the cross validation
            can select the needed configurations
        folds: the number of folds the user requires for cross-validation
        optimizer: the optimizer the user wants to
            run like Adam for the model's backprop

        Returns
        -------
        return: tuple pairs of metrics with optimal configuration.
        """

        # we save a model configuration object that we can then
        # disperse through the master protocol
        model_config = {
            "model": model,
            "model_params": parameters,
            "optimizer": optimizer,
            "epochs": epochs,
        }

        # we call our master class that will allow
        # use to schedule workers and initialize jobs
        master = Master(
            dataset,
            k_folds=folds,
            batch_size=batch_size,
            use_gpu=use_gpu,
            backend=backend,
            num_workers=num_workers,
        )

        # we can then call the master backend to run our search
        return master.run(model_config)
