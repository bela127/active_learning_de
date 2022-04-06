from __future__ import annotations
from typing import TYPE_CHECKING

from random import choice

import numpy as np

from ide.core.data_pool import DataPool
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Dict

class FlatDataPool(DataPool):
    """
    implements a pool of already labeled data
    """

    iteration = 0

    def __init__(self, query_shape, result_shape):
        super().__init__(query_shape, result_shape)
        self.query_index: Dict = {}

    
    def query(self, queries):
        result_list = []
        for query in queries:
            result_candidate = self.query_index.get(tuple(query), [])
            result = choice(result_candidate)
            result_list.append(result)
        
        results: np.ndarray = np.asarray(result_list)
        return queries, results


    def add(self, data_points):
        super().add(data_points)

        queries, results = data_points

        for query, result in zip(queries, results):

            results = self.query_index.get(tuple(query), [])
            self.query_index[tuple(query)] = results + [result]

    @property
    def query_pool(self) -> QueryPool:
        return QueryPool(self.queries.size, self._oracle_query_pool.query_shape, self._oracle_query_pool.query_ranges)