from typing import Dict
from typing import List
from ray.train import TrainingCallback

class MetricsCallback(TrainingCallback):
    def handle_result(self, results: List[Dict], **info):
        print(results)
