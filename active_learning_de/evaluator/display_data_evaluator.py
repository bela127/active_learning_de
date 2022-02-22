from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class DisplayDataEvaluator(EvaluationMetric):
    """
    Evaluation Metric. Evaluates the average time required for training. Evaluation a list of the average round times

    e.g. evaluation: [avg time for round 1, avg time for round 1-2, ..., avg time for round 1-n]
    """
    def __init__(self):
        self.round_number = -1
        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        x, y = [],[]
        self.sc = ax.scatter(x,y)
        plt.xlim(0,1)
        plt.ylim(-1,1)

        plt.draw()
    
    def signal_round_stop(self):
        self.round_number += 1

    def signal_knowledge_discovery_stop(self):
        query = tf.convert_to_tensor(list(range(0,len(self.blueprint.knowledge_discovery_task.sampler.pool.get_all_elements()))))
        xs, ys = self.blueprint.knowledge_discovery_task.surrogate_model.query(query)
        

        self.sc.set_offsets(np.c_[xs,ys])
        self.fig.canvas.draw_idle()
        plt.pause(0.02)

        

    def get_evaluation(self):
        plt.waitforbuttonpress()
        
