from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from scipy.stats import kruskal

from active_learning_de.new_api.building_blocks.data_sampler import DataSampler
from active_learning_de.new_api.core.configuration import Configurable

if TYPE_CHECKING:
    from active_learning_de.new_api.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class MultiSampleTest(Configurable):

    data_sampler: DataSampler
    
    def query(self, queries):
        query1, query2 = queries
        sample1 = self.data_sampler.sample(query1)
        sample2 = self.data_sampler.sample(query2)

        t,p = self.test(sample1, sample2)
        return t,p
    
    def test(self, sample1: Tuple[float,...], sample2: Tuple[float,...]) -> Tuple[List[float],List[float]]:
        ...


@dataclass
class KWHMultiSampleTest(MultiSampleTest):
    
    
    def test(self, sample1, sample2):
        t, p = kruskal(sample1, sample2)
        return t, p