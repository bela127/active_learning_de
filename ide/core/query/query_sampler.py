from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.configuration import Configurable


if TYPE_CHECKING:
    from ide.core.query.query_pool import QueryPool
    from typing import Optional
    from nptyping import NDArray, Number, Shape


@dataclass
class QuerySampler(Configurable):
    num_queries: int = 1

    def sample(self, query_pool: QueryPool, num_queries: Optional[int] = None) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        raise NotImplementedError

    @property
    def query_pool(self) -> QueryPool:
        raise NotImplementedError