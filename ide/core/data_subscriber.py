from __future__ import annotations
from typing import TYPE_CHECKING

from ide.core.data_pool import DataPool
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple
    from typing_extensions import Self #type: ignore
    from nptyping import NDArray, Number, Shape

class DataSubscriber(Configurable):

    def update(self, data_points: Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]):
        raise NotImplementedError
    
    def __call__(self, data_pool: DataPool = None, **kwargs) -> Self:
        obj = super().__call__( **kwargs)

        obj.data_pool = data_pool
        if isinstance(data_pool, DataPool):
            data_pool.subscribe(obj)

        return obj
