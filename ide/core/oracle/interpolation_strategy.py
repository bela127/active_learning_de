from __future__ import annotations
from typing import TYPE_CHECKING
from ide.building_blocks.data_sampler import DataSampler



from ide.core.configuration import Configurable
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    import nptyping as npt

class InterpolationStrategy(Configurable):
    data_sampler: DataSampler

    def interpolate(self, data_points) -> Tuple[npt.NDArray, npt.NDArray]:
        return data_points

    @property
    def query_pool(self) -> QueryPool:
        return self.data_sampler.query_pool

class NoInterpolation(InterpolationStrategy):
    ...
