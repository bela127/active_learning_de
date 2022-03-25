from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from ide.building_blocks.data_sampler import DataSampler
from ide.building_blocks.two_sample_test import TwoSampleTest
from ide.core.experiment_module import ExperimentModule
from ide.core.queryable import Queryable

if TYPE_CHECKING:
    from typing import Tuple, List, Any
    from typing_extensions import Self
    from ide.core.experiment_modules import ExperimentModules


@dataclass
class TestInterpolator(Queryable, ExperimentModule):
    test: TwoSampleTest

@dataclass
class KNNTestInterpolator(TestInterpolator):
    data_sampler: DataSampler
    
    def query(self, queries):

        queries1 = queries[:,0,:]
        queries2 = queries[:,0,:]

        sample_queries1, sample1 = self.data_sampler.sample(queries1)
        sample_queries2, sample2 = self.data_sampler.sample(queries2)

        t,p = self.test.test(sample1, sample2)

        u = self.uncertainty((queries1, queries2), sample_queries1, sample_queries2)

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

    def __call__(self, exp_modules: ExperimentModules, *args, **kwargs) -> Self:
        obj: Self = super().__call__(exp_modules, *args, **kwargs)
        obj.data_sampler = obj.data_sampler(exp_modules.data_pool)
        return obj
