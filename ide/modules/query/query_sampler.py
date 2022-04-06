from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from ide.core.query.query_sampler import QuerySampler

if TYPE_CHECKING:
    from typing import Tuple, List, Union

@dataclass
class RandomQuerySampler(QuerySampler):

    def sample(self, query_pool, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if query_pool.query_count:
            count = query_pool.query_count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            return np.random.randint(low = 0, high = count, size=(num_queries,))
        else:
            a = query_pool.elements_from_norm_pos(np.random.uniform(size=(num_queries, *query_pool.query_shape)))
            return a

class LatinHypercubeSampler(QuerySampler):

    def sample(self, query_pool, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        dim = 1
        for size in query_pool.query_shape:
            dim *= size

        sampler = qmc.LatinHypercube(d=dim)
        
        if query_pool.query_count:
            count = query_pool.query_count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            
            return np.random.randint(low = 0, high = count, size=(num_queries,))
        else:
            sample = sampler.random(n=num_queries)
            
            sample = np.reshape(sample, (num_queries, *query_pool.query_shape))

            a = query_pool.elements_from_norm_pos(sample)
            return a