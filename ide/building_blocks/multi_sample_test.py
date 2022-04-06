from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from scipy.stats import kruskal #type: ignore

from ide.core.data_sampler import DataSampler
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from ide.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class MultiSampleTest(Configurable):


    def test(self, samples: Tuple[float,...]) -> Tuple[List[float],List[float]]:
        raise NotImplementedError


@dataclass
class KWHMultiSampleTest(MultiSampleTest):
    
    
    def test(self, samples):
        t, p = kruskal(*samples)
        return t, p