from typing import Tuple, Union

import tensorflow as tf

from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.data_retrievement.pools.discrete_vector_pool import DiscreteVectorPool
from active_learning_ts.data_retrievement.retrievement_strategies.nearest_neighbours_retreivement_strategy import (
    NearestNeighboursRetrievementStrategy,
)
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import (
    ExactRetrievement,
)


class PoolSurrogateModel(SurrogateModel):
    """
    implements a pool of already labeled data
    """

    def __init__(self):
        self.in_dim: int = None
        self.training_points: tf.Tensor = None
        self.training_values: tf.Tensor = None

    def post_init(self, data_retriever: DataRetriever):
        super().post_init(data_retriever)
        self.in_dim = self.query_pool.shape[0]
        self.query_pool = DiscreteVectorPool(
            self.in_dim, [], ExactRetrievement()
        )

    def learn(self, points: tf.Tensor, values: tf.Tensor):
        if self.training_points is None:
            self.training_points = points
            self.training_values = values
        else:
            self.training_points = tf.concat([self.training_points, points], 0)
            self.training_values = tf.concat([self.training_values, values], 0)
        self.query_pool = DiscreteVectorPool(
            self.in_dim, self.training_points, ExactRetrievement()
        )

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor([0.0] * len(points))

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.gather(self.training_points, points)
        y = tf.gather(self.training_values, points)
        return x, y
