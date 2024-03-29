from typing import Dict
from typing import List
from ray.train import TrainingCallback


class MetricsCallback(TrainingCallback):
    """
    Class for defining behavior of callback logging structure.
    """

    def handle_result(self, results: List[Dict], **info):
        """
        Callback function for parsing the object details given by each worker

        Params
        ------
        results: logs results of metrics and model details / and
        state after each iteration

        Returns
        -------
        None
        """
        for result in results:
            print(f"Status for Worker: {result['worker_id']}")
            print('[Epoch %d] average train loss: %.3f' %
                  (result["epoch"] + 1, result["train_loss"]))
            print('[Epoch %d] average validation loss: %.3f' %
                  (result["epoch"] + 1, result["val_loss"]))
            print('[Epoch %d] average validation accuracy: %.3f' %
                  (result["epoch"] + 1, result["val_accuracy"]))
