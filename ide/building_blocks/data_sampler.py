from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field


from rtree import index
from rtree.index import RT_Memory
import numpy as np

from ide.core.configuration import Configurable
from ide.building_blocks.data_pool import DataPool

if TYPE_CHECKING:
    from typing import Tuple
    from typing_extensions import Self

@dataclass
class DataSampler(Configurable):

    sample_size: int
    data_pool: DataPool = field(init=False)

    #@abstractmethod
    def sample(self, query, size = None) -> Tuple[Tuple[float, ...],Tuple[float, ...]]:
        ...

    @property
    def query_pool(self):
        self.data_pool.query_pool

    def __call__(self, data_pool, *args, **kwargs) -> Self:
        obj = super().__call__(*args, **kwargs)

        obj.data_pool = data_pool
        data_pool.subscrib(obj)

        return obj

@dataclass
class KNNDataSampler(DataSampler):

    def __init__(self, sample_size, shape):
        super().__init__(sample_size)
        p = index.Property()

        dimension = 1
        for size in shape:
            dimension *= dimension*size

        p.dimension = dimension
        p.storage = RT_Memory

        self._r_tree = index.Index(properties=p)
        self._points = {}

    def update(self, data_point: Tuple[np.ndarray, np.ndarray]):
        query, result = data_point

        
        index = np.concatenate((query.flatten(),query.flatten()))

        oid = id(result)

        self._points[oid] = query, result
        self._r_tree.insert(oid, index)

    def sample(self, query, size = None):
        if size is None: size = self.sample_size


        index = np.concatenate((query.flatten(),query.flatten()))

        ids = self._r_tree.nearest(index, size)
        queries = []
        results = []
        for oid in ids:
            query, result = self._points[oid]
            queries.append(query)
            results.append(result)
        
        data_points = (queries, results)
        return data_points
