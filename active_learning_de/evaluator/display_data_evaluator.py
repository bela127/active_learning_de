from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
import matplotlib.pyplot as plt
import tensorflow as tf

class DisplayDataEvaluator(EvaluationMetric):
    """
    Evaluation Metric. Evaluates the average time required for training. Evaluation a list of the average round times

    e.g. evaluation: [avg time for round 1, avg time for round 1-2, ..., avg time for round 1-n]
    """

    def signal_end_of_experiment(self):
        query = tf.convert_to_tensor(list(range(0,len(self.blueprint.knowledge_discovery_task.sampler.pool.get_all_elements()))))
        xs, ys = self.blueprint.knowledge_discovery_task.surrogate_model.query(query)
        
        plt.scatter(xs, ys)
        plt.show()
