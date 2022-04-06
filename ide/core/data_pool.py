from __future__ import annotations
from typing import TYPE_CHECKING, NewType

import numpy as np

from ide.core.queryable import Queryable
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from typing import Tuple, List, Dict

    from nptyping import  NDArray, Number, Shape

    from ide.core.data_subscriber import DataSubscriber

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
        
    def __call__(self, oracle_query_pool: QueryPool = None, **kwargs) -> Self:
        obj = super().__call__( **kwargs)
        obj._oracle_query_pool = oracle_query_pool #TODO need more infos like shape
        return obj
