from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from ide.core.configuration import Configurable
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List, Union

@dataclass
class QuerySampler(Configurable):
    num_queries: int = 1

    def sample(self, pool, num_queries: Union[int, None] = None) -> Tuple[float,...]:
        ...

    @property
    def query_pool(self) -> QueryPool:
        ...

@dataclass
class RandomQuerySampler(QuerySampler):

    def sample(self, query_pool: QueryPool, num_queries: Union[int, None] = None):
        if num_queries is None: num_queries = self.num_queries
        
        if query_pool.query_count:
            count = query_pool.query_count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            return np.random.randint(low = 0, high = count, size=(num_queries,))
        else:
            a = query_pool.elements_from_norm_pos(np.random.uniform(size=(num_queries, *query_pool.query_shape)))
            return a