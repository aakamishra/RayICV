from master import Master


class RayCrossValidation:
    def __init__(self, model, dataset, folds, optimizer, epochs=10):
        model_config = {
            "model": model,
            "optimizer": optimizer,
            "epochs": epochs,
        }
        master = Master(dataset, k_folds=5, batch_size=32, use_gpu=False,
                        backend="torch")
        return master.run(model_config)
