from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod, abstractproperty


from ide.core.configuration import Configurable


if TYPE_CHECKING:
    from typing import Tuple, List
    from nptyping import NDArray, Shape, Number

    from ide.core.query.query_pool import QueryPool
    from ide.core.data.data_pool import DataPool

class Queryable(Configurable):

    @abstractmethod
    def query(self, queries: NDArray[Number,  Shape["query_nr, ... query_shape"]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_shape"]], NDArray[Number, Shape["query_nr, ... result_shape"]]]:
        raise NotImplementedError


    @abstractproperty
    def query_pool(self) -> QueryPool:
        raise NotImplementedError
    
    @abstractproperty
    def data_pool(self) -> DataPool:
        raise NotImplementedError