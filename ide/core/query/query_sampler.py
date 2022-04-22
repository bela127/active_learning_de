from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

from ide.core.configuration import Configurable
from ide.core.queryable import Queryable



if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from ide.core.query.query_pool import QueryPool
    from typing import Optional
    from nptyping import NDArray, Number, Shape


@dataclass
class QuerySampler(Configurable):
    num_queries: int = 1

    queryable: Queryable = field(init=False, default=None)

    @abstractmethod
    def sample(self, num_queries: Optional[int] = None) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        raise NotImplementedError

    @property
    def query_pool(self) -> QueryPool:
        return self.queryable.query_pool

    def __call__(self, queryable: Queryable = None, **kwargs) -> Self:
        obj = super().__call__(**kwargs)
        if isinstance(queryable, Queryable):
            obj.queryable = queryable
        else:
            raise ValueError
        return obj