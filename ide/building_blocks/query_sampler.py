from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class QuerySampler(Configurable):

    def sample(self, pool, num_queries: int = 1) -> Tuple[float,...]:
        ...

@dataclass
class RandomQuerySampler(QuerySampler):

    def sample(self, pool, num_queries: int = 1):
        if pool.is_discrete:
            count = pool.query_count
            if count == 0:
                return np.asarray([], dtype=np.int32)
            return np.random.randint(low = 0, high = count, size=(num_queries,))
        else:
            a = pool.get_elements_normalized(np.random.uniform(size=(num_queries, self.pool.shape[0])))
            a = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
        return a