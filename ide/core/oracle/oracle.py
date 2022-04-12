from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.oracle.data_source import DataSource
from ide.core.oracle.augmentation import Augmentation
from ide.core.query.query_pool import QueryPool
from ide.core.queryable import Queryable


if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from typing import Tuple
    from nptyping import NDArray, Number, Shape


@dataclass
class Oracle(Queryable):
    """
    Uses the given retrievement strategy in order to retrieve data from the given data source
    """
    data_source: DataSource
    augmentation: Augmentation


    def query(self, query_candidates: NDArray[Number, Shape["query_nr, ... query_dim"]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        data_points = self.data_source.query(query_candidates)
        augmented_data_points = self.augmentation.apply(data_points)

        return augmented_data_points

    @property
    def query_pool(self):
        return self.data_source.query_pool

    @property
    def data_pool(self):
        return self.data_source.data_pool

    def __call__(self, *args, **kwargs) -> Self:
        obj = super().__call__(*args, **kwargs)

        obj.data_source = obj.data_source()
        obj.augmentation = obj.augmentation()

        return obj



