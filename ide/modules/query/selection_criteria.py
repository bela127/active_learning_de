from __future__ import annotations
import imp
from typing import TYPE_CHECKING

from ide.core.query.selection_criteria import SelectionCriteria

if TYPE_CHECKING:
    from nptyping import NDArray, Shape, Number
    from ide.core.query.query_pool import QueryPool

class NoSelectionCriteria(SelectionCriteria):

    def query(self, queries: NDArray[Number,  Shape["query_nr, ... query_shape"]]) -> float:
        raise NameError("query(queries), should never be called on NoSelectionCriteria")

    @property
    def query_pool(self) -> QueryPool:
        raise NotImplementedError