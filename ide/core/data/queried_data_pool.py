from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ide.core.queryable import Queryable

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from typing import Tuple, List, Dict

    from nptyping import  NDArray, Number, Shape

    from ide.core.data_subscriber import DataSubscriber
    from ide.core.query.query_pool import QueryPool
    from ide.core.data.data_pool import DataPool


class QueriedDataPool(Queryable):
    _oracle_query_pool: QueryPool
    _oracle_data_pool: DataPool

    def __init__(self):
        self._subscriber: List[DataSubscriber] = []

        self.queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None
        self.results: NDArray[Number, Shape["query_nr, ... result_dim"]] = None

        self.last_queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None

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
        
        self.last_queries = queries

        for subscriber in self._subscriber:
            subscriber.update(data_points)
        
    def __call__(self, oracle_query_pool: QueryPool = None, oracle_data_pool: QueryPool = None, **kwargs) -> Self:
        obj = super().__call__( **kwargs)
        obj._oracle_query_pool = oracle_query_pool
        obj._oracle_data_pool = oracle_data_pool

        obj.queries = np.empty((0,*obj._oracle_query_pool.query_shape))
        obj.results = np.empty((0,*obj._oracle_data_pool.result_shape))

        obj.last_queries = np.empty((0,*obj._oracle_query_pool.query_shape))


        return obj
    
