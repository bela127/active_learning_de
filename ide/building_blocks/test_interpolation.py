from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from ide.building_blocks.data_sampler import DataSampler
from ide.building_blocks.two_sample_test import TwoSampleTest
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class KNNTestInterpolator(Configurable):
    data_sampler: DataSampler
    test: TwoSampleTest
    
    def query(self, queries):
        query1, query2 = queries
        queries1, sample1 = self.data_sampler.sample(query1)
        queries2, sample2 = self.data_sampler.sample(query2)

        t,p = self.test.test(sample1, sample2)

        u = self.uncertainty(queries, queries1, queries2)

        return t, p, u
    
    def uncertainty(self, queries, queries1, queries2):
        query1, query2 = queries
        dists1 = np.linalg.norm(queries1-query1, axis=1)
        dists2 = np.linalg.norm(queries2-query2, axis=1)
        mean_dist = np.mean(np.concatenate((dists1,dists2)))
        return mean_dist