from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from ide.core.configuration import Configurable
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List
    from nptyping import NDArray, Shape, Number

class Queryable(Configurable):

    def query(self, queries: NDArray[Number,  Shape["query_nr, ..."]]) -> Tuple[NDArray[Number, Shape["query_nr, ..."]], NDArray[Number, Shape["query_nr, ..."]]]:
        ...

    @property
    def query_pool(self) -> QueryPool:
        ...