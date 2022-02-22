import time

from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric


class SimpleResultEvaluator(EvaluationMetric):
    """
    Evaluation Metric. Evaluates the average time required for training. Evaluation a list of the average round times

    e.g. evaluation: [avg time for round 1, avg time for round 1-2, ..., avg time for round 1-n]
    """
    def __init__(self):
        self.round_number = -1
        self.results = []
    
    def signal_round_stop(self):
        self.round_number += 1

    def signal_knowledge_discovery_stop(self):
        result = self.blueprint.knowledge_discovery_task.learn(100)
        self.results.append(result)

    def eval(self):
        print(self.results[self.round_number])

    def get_evaluation(self):
        return self.results
