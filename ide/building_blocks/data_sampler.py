from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field


from rtree import index
from rtree.index import RT_Memory
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ide.core.configuration import Configurable
from ide.building_blocks.data_pool import DataPool
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple
    from typing_extensions import Self
    from nptyping import NDArray, Number, Shape

@dataclass
class DataSampler(Configurable):

    sample_size: int
    data_pool: DataPool = field(init=False)

    #@abstractmethod
    def sample(self, queries: NDArray[Number, Shape["query_nr, ... query_dim"]], size = None) -> Tuple[NDArray[Number, Shape["query_nr, sample_size, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_size,... result_dim"]]]:
        raise NotImplementedError()
    
    def update(self, data_points: Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]):
        raise NotImplementedError()


    @property
    def query_pool(self):
        self.data_pool.query_pool

    def __call__(self, data_pool, *args, **kwargs) -> Self:
        obj = super().__call__(*args, **kwargs)

        obj.data_pool = data_pool
        data_pool.subscribe(obj)

        return obj


@dataclass
class RTreeKNNDataSampler(DataSampler):

    def __init__(self, sample_size, query_shape):
        super().__init__(sample_size)
        p = index.Property()

        dimension = 1
        for size in query_shape:
            dimension *= dimension*size

        p.dimension = dimension
        p.storage = RT_Memory

        self._r_tree = index.Index(stream = [], properties=p)
        self._points = {}

    def update(self, data_points: Tuple[np.ndarray, np.ndarray]):
        queries, results = data_points

        for query, result in zip(queries, results):        
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
    
    @property
    def query_pool(self):
        bounds = self._r_tree.get_bounds(False)
        query_count = self._r_tree.get_size()

        return QueryPool(False, query_shape=self.data_pool.query_pool.query_shape)


@dataclass
class KDTreeKNNDataSampler(DataSampler):

    def __init__(self, sample_size):
        super().__init__(sample_size)
        self._knn = NearestNeighbors(n_neighbors=self.sample_size)
        

    def update(self, data_points):
        self._knn.fit(self.data_pool.queries, self.data_pool.results)

    def sample(self, queries, size = None):
        if size is None: size = self.sample_size
        if self.data_pool.query_pool.query_count < size: size = self.data_pool.query_pool.query_count

        kneighbor_indexes = self._knn.kneighbors(queries, n_neighbors=size, return_distance=False)

        neighbor_queries = self.data_pool.queries[kneighbor_indexes]
        kneighbors = self.data_pool.results[kneighbor_indexes]
        
        return (neighbor_queries, kneighbors)
    
    @property
    def query_pool(self):
        return QueryPool(query_count=None, query_shape=self.data_pool.query_pool.query_shape, query_ranges=self.data_pool.query_pool.query_ranges)