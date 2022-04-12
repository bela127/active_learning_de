from __future__ import annotations
from typing import TYPE_CHECKING

from random import choice

import numpy as np
from ide.core.data.data_pool import DataPool

from ide.core.data.queried_data_pool import QueriedDataPool
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Dict

class FlatQueriedDataPool(QueriedDataPool):
    """
    implements a pool of already labeled data
    """

    iteration = 0

    def __init__(self):
        super().__init__()
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
    def query_pool(self):
        query_pool = QueryPool(query_count = self.queries.size, query_shape = self._oracle_query_pool.query_shape, query_ranges= None)
        query_pool._queries = self.queries
        query_pool._last_queries = self.last_queries
        return query_pool

    @property
    def data_pool(self):
        return DataPool(self.query_pool, self._oracle_data_pool.result_shape)