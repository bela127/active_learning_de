from __future__ import annotations
from abc import abstractproperty
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
from ide.core.data.data_pool import DataPool

from ide.core.experiment_module import ExperimentModule
from ide.core.queryable import Queryable

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from ide.core.experiment_modules import ExperimentModules
    from ide.core.data_sampler import DataSampler
    from ide.building_blocks.two_sample_test import TwoSampleTest



@dataclass
class TestInterpolator(Queryable, ExperimentModule):
    test: TwoSampleTest

    def __call__(self, exp_modules: ExperimentModules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.test = obj.test()
        return obj

@dataclass
class KNNTestInterpolator(TestInterpolator):
    data_sampler: DataSampler
    
    def query(self, queries):

        queries1 = queries[:,0,:]
        queries2 = queries[:,1,:]

        sample_queries1, samples1 = self.data_sampler.sample(queries1)
        sample_queries2, samples2 = self.data_sampler.sample(queries2)

        t,p = self.test.test(samples1, samples2)

        u = self.uncertainty((queries1, queries2), sample_queries1, sample_queries2)
        #u = 0

        return t, p, u
    
    def uncertainty(self, queries, queries1, queries2):
        query1, query2 = queries
        dists1 = np.linalg.norm(queries1-query1[:,None,:], axis=2)
        dists2 = np.linalg.norm(queries2-query2[:,None,:], axis=2)
        mean_dist = np.mean(np.concatenate((dists1,dists2), axis=1),axis=1)
        return mean_dist

    @property
    def query_pool(self):
        return self.data_sampler.query_pool

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, result_shape=(3,))

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.data_sampler = obj.data_sampler(exp_modules)
        return obj
