from master import Master

class RayCrossValidation:
	def __init__(self, model, dataset, folds, optimizer, epochs=10):
		model_config = {"model" : model, "optimizer" : optimizer, "epochs" : epochs}
		return Master(dataset, k_folds=5, batch_size=32, use_gpu=False, backend="torch").run(model_config)