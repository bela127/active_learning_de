from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
import tensorflow as tf #type: ignore


from ide.core.query.query_optimizer import QueryOptimizer

if TYPE_CHECKING:
    from ide.core.query.query_sampler import QuerySampler


@dataclass
class MCQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler
    num_tries: int

    
class MaxMCQueryOptimizer(MCQueryOptimizer):

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        query_candidates = self.query_sampler.sample(self.selection_criteria.query_pool, self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        queries = tf.gather(query_candidates, tf.math.top_k(scores, num_queries).indices)

        queries = queries.numpy()

        return queries

@dataclass
class ProbWeightedMCQueryOptimizer(MCQueryOptimizer):
    _rng = np.random.default_rng()

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        query_candidates = self.query_sampler.sample(self.selection_criteria.pool, self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        queries = self._rng.choice(a=query_candidates, size=num_queries, replace=False, p=scores)

        return queries