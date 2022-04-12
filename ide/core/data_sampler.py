from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import TYPE_CHECKING

from dataclasses import dataclass, field


from ide.core.data_subscriber import DataSubscriber


if TYPE_CHECKING:
    from typing import Tuple
    from typing_extensions import Self #type: ignore
    from nptyping import NDArray, Number, Shape
    from ide.core.data.queried_data_pool import QueriedDataPool

class DataSampler(DataSubscriber):

    @abstractmethod
    def sample(self, queries: NDArray[Number, Shape["query_nr, ... query_dim"]], size = None) -> Tuple[NDArray[Number, Shape["query_nr, sample_size, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_size,... result_dim"]]]:
        raise NotImplementedError()


    @abstractproperty
    def query_pool(self):
        raise NotImplementedError()