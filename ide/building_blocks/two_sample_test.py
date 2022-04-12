from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

from scipy.stats import mannwhitneyu #type: ignore

from ide.core.data_sampler import DataSampler
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from ide.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class TwoSampleTest(Configurable):
    
    def test(self, sample1: Tuple[float,...], sample2: Tuple[float,...]) -> Tuple[List[float],List[float]]:
        ...


@dataclass
class MWUTwoSampleTest(TwoSampleTest):
    
    def test(self, samples1, samples2):

        #def step(i):
        #    return mannwhitneyu(samples1[i], samples2[i])
        
        #idx = np.arange(len(samples1))
        #results = np.array(list(map(step, idx)))

        U, p = mannwhitneyu(samples1, samples2, method="exact", axis=1)
        #U = results[:,0,:]
        #p = results[:,1,:]
        return U, p

        
    
