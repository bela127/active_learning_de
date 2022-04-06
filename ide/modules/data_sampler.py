from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field


from rtree import index #type: ignore
from rtree.index import RT_Memory #type: ignore
import numpy as np
from sklearn.neighbors import NearestNeighbors #type: ignore

from ide.core.data_sampler import DataSampler
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple

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

    def __init__(self, sample_size, sample_size_data_fraction = 4):
        super().__init__(sample_size)
        self.sample_size_data_fraction = sample_size_data_fraction
        self._knn = NearestNeighbors(n_neighbors=self.sample_size)
        

    def update(self, data_points):
        self._knn.fit(self.data_pool.queries, self.data_pool.results)

    def sample(self, queries, size = None):
        if size is None: size = self.sample_size
        if self.data_pool.query_pool.query_count // self.sample_size_data_fraction < size: size = self.data_pool.query_pool.query_count // self.sample_size_data_fraction

        kneighbor_indexes = self._knn.kneighbors(queries, n_neighbors=size, return_distance=False)

        neighbor_queries = self.data_pool.queries[kneighbor_indexes]
        kneighbors = self.data_pool.results[kneighbor_indexes]
        
        return (neighbor_queries, kneighbors)
    
    @property
    def query_pool(self):
        return QueryPool(query_count=None, query_shape=self.data_pool.query_pool.query_shape, query_ranges=self.data_pool.query_pool.query_ranges)