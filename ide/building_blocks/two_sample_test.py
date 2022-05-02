from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

from scipy.stats import mannwhitneyu #type: ignore
from scipy.stats import kruskal #type: ignore


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

        U1, p = mannwhitneyu(samples1, samples2, method="exact", axis=1)
        U2 = samples1.shape[1] * samples2.shape[1] - U1
        U = np.min((U1,U2), axis=0)
        r = 1 - (2*U)/(samples1.shape[1] * samples2.shape[1])

        #U = results[:,0,:]
        #p = results[:,1,:]
        return r, p

@dataclass
class KWHTwoSampleTest(TwoSampleTest):
    
    def test(self, samples1, samples2):

        #def step(i):
        #    return mannwhitneyu(samples1[i], samples2[i])
        
        #idx = np.arange(len(samples1))
        #results = np.array(list(map(step, idx)))

        t, p = kruskal(samples1, samples2, axis=1)

        #U = results[:,0,:]
        #p = results[:,1,:]
        return t, p

        
    
