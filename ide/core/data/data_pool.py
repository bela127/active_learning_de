from __future__ import annotations
from typing import TYPE_CHECKING, NewType

from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple

class DataPool(QueryPool):

    def __init__(self, query_pool: QueryPool, result_shape: Tuple[int,...]):
        super().__init__(query_count=query_pool.query_count, query_shape=query_pool.query_shape, query_ranges=query_pool.query_ranges)
        self._queries = query_pool._queries
        self._last_queries = query_pool._last_queries

        self.result_shape: Tuple[int,...] = result_shape
