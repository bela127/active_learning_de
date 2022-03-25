from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from ide.core.queryable import Queryable
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List
    from nptyping import NDArray, Shape, Number

@dataclass
class DataSource(Queryable):

    query_shape: Tuple[int,...]
    result_shape: Tuple[int,...]

    def query(self, queries: NDArray[Number, Shape["query_nr, ... query_dim"]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        ...

    @property
    def query_pool(self) -> QueryPool:
        ...