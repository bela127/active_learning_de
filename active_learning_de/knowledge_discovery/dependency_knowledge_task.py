import numpy
import tensorflow as tf
from scipy.stats import pearsonr

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class DependencyKnowledgeTask(KnowledgeDiscoveryTask):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def learn(self, num_queries):
        xs = self.sampler.sample(num_queries)
        ys = self.surrogate_model.query(xs)

        r, p = pearsonr(xs, ys)
        self.global_uncertainty = p

        return r, p


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points[0].shape, self.global_uncertainty)
