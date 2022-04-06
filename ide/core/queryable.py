from __future__ import annotations
from typing import TYPE_CHECKING

from ide.core.configuration import Configurable
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List
    from nptyping import NDArray, Shape, Number

class Queryable(Configurable):

    def query(self, queries: NDArray[Number,  Shape["query_nr, ... query_shape"]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_shape"]], NDArray[Number, Shape["query_nr, ... result_shape"]]]:
        raise NotImplementedError

    @property
    def query_pool(self) -> QueryPool:
        raise NotImplementedError