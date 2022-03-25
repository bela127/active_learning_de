from __future__ import annotations
from typing import TYPE_CHECKING

from random import choice

import numpy as np

from ide.core.queryable import Queryable
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Tuple, List, Dict

    from nptyping import NDArray, Number, Shape

    from ide.building_blocks.data_subscriber import DataSubscriber

class DataPool(Queryable):
    _oracle_query_pool: QueryPool

    def __init__(self, query_shape, result_shape):
        self.query_shape: Tuple[int,...] = query_shape
        self.result_shape: Tuple[int,...] = result_shape
        
        self._subscriber: List[DataSubscriber] = []

        self.queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None
        self.results: NDArray[Number, Shape["query_nr, ... result_dim"]] = None

    def subscribe(self, subscriber):
        self._subscriber.append(subscriber)


    def add(self, data_points: Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]):
        queries, results = data_points

        if self.queries is None:
            self.queries = queries
            self.results = results
        else:
            self.queries = np.concatenate((self.queries, queries))
            self.results = np.concatenate((self.results, results))

        for subscriber in self._subscriber:
            subscriber.update(data_points)
        
    def __call__(self, oracle_query_pool, *args, **kwargs) -> Self:
        obj = super().__call__(*args, **kwargs)
        obj._oracle_query_pool = oracle_query_pool #TODO need more infos like shape
        return obj

class FlatDataPool(DataPool):
    """
    implements a pool of already labeled data
    """
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