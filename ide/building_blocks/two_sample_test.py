from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from scipy.stats import mannwhitneyu
import numpy as np

from ide.building_blocks.data_sampler import DataSampler
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from ide.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class TwoSampleTest(Configurable):

    data_sampler: DataSampler
    
    def query(self, queries):
        query1, query2 = queries
        queries1, sample1 = self.data_sampler.sample(query1)
        queries2, sample2 = self.data_sampler.sample(query2)

        t,p = self.test(sample1, sample2)

        return t, p
    
    def test(self, sample1: Tuple[float,...], sample2: Tuple[float,...]) -> Tuple[List[float],List[float]]:
        ...



@dataclass
class MWUTwoSampleTest(TwoSampleTest):
    
    
    def test(self, sample1, sample2):
        U, p = mannwhitneyu(sample1, sample2, method="exact")
        return U, p

        
    
