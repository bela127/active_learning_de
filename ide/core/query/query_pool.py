from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    from typing import Tuple, List, Union, Optional
    from nptyping import NDArray, Number, Shape


@dataclass
class QueryPool():

    query_count: Optional[int]
    query_shape: Tuple[int, ...]
    query_ranges: Optional[NDArray[Number, Shape["... query_dims,[xi_min, xi_max]"]]]

    _queries: Optional[NDArray[Number, Shape["query_count, ... query_shape"]]] = field(init=False, default=None)
    _last_queries: Optional[NDArray[Number, Shape["query_count, ... query_shape"]]] = field(init=False, default=None)


    def add_queries(self, queries):
        if self._queries is None:
            self._queries = queries
        else: 
            self._queries = np.concatenate((self._queries, queries))
        self._last_queries = queries
        self.query_count = self._queries.shape[0]

    def last_queries(self):
        if self._last_queries is None:
            raise LookupError("there are infinit queries continues pool")
        return self._last_queries

    def queries_from_norm_pos(self, norm_pos: NDArray[Number, Shape["query_nr, ... query_dims"]]) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        if self.query_ranges is None:
            raise LookupError("can not look up a position in a discrete pool")
        elements = self.query_ranges[..., 0] + (self.query_ranges[..., 1] - self.query_ranges[..., 0]) * norm_pos
        return elements
    
    def queries_from_index(self, indexes):
        if self._queries is None:
            raise LookupError("can not look up a index in a continues pool")
        return self._queries[indexes]
    
    def all_queries(self):
        if self._queries is None:
            raise LookupError("there are infinit queries continues pool")
        return self._queries