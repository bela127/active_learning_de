from __future__ import annotations
from typing import TYPE_CHECKING

from math import ceil
import random
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc # type: ignore

from ide.core.query.query_sampler import QuerySampler

if TYPE_CHECKING:
    from typing import Tuple, List, Union
    from nptyping import NDArray, Number, Shape

@dataclass
class OptimalQuerySampler(QuerySampler):
    optimal_queries: Tuple[NDArray[Number, Shape["query_nr, ... query_dims"]], ...] = None

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if self.query_pool.query_ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            query_nr = self.optimal_queries[0].shape[0]
            k = ceil(num_queries / query_nr) 
            queries = random.choices(self.optimal_queries, k=k)
            queries = np.concatenate(queries)
            return queries[:num_queries]

@dataclass
class UniformQuerySampler(QuerySampler):

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if self.query_pool.query_ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            a = self.query_pool.queries_from_norm_pos(np.random.uniform(size=(num_queries, *self.query_pool.query_shape)))
            return a

class LatinHypercubeQuerySampler(QuerySampler):

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        dim = 1
        for size in self.query_pool.query_shape:
            dim *= size

        sampler = qmc.LatinHypercube(d=dim)
        
        if self.query_pool.query_ranges is None:
            raise ValueError("Not for discrete Pools")
        else:
            sample = sampler.random(n=num_queries)
            
            sample = np.reshape(sample, (num_queries, *self.query_pool.query_shape))

            a = self.query_pool.queries_from_norm_pos(sample)
            return a

class RandomChoiceQuerySampler(QuerySampler):

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if self.query_pool.query_count is None:
            raise ValueError("Not for continues pools")
        else:
            count = self.query_pool.query_count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            return self.query_pool.queries_from_index(np.random.randint(low = 0, high = count, size=(num_queries,)))
@dataclass
class LastQuerySampler(QuerySampler):
    num_queries: int = None

    def sample(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        
        if self.query_pool.query_count is None:
            raise ValueError("Not for continues pools")
        else:
            return self.query_pool.last_queries()