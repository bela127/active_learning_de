from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class DynamicDisplayDataEvaluator(EvaluationMetric):
    """
    Evaluation Metric. Evaluates the average time required for training. Evaluation a list of the average round times

    e.g. evaluation: [avg time for round 1, avg time for round 1-2, ..., avg time for round 1-n]
    """
    def __init__(self):
        self.round_number = -1
        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax 
        x, y = [],[]
        self.sc = ax.scatter(x,y)

        plt.draw()
    
    def signal_round_stop(self):
        self.round_number += 1

        #TODO signal_experiment_end
    def signal_knowledge_discovery_stop(self):
        #TODO discrete pool should directly have a property for geting all querys (which are the indexes)
        query = tf.convert_to_tensor(list(range(0,len(self.blueprint.knowledge_discovery_task.sampler.pool.get_all_elements()))))
        xs, ys = self.blueprint.knowledge_discovery_task.surrogate_model.query(query)
        
        self.ax.set_xlim(min(xs),max(xs))
        self.ax.set_ylim(min(ys),max(ys))
        self.sc.set_offsets(np.c_[xs,ys])
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def get_evaluation(self):
        plt.waitforbuttonpress()
        
