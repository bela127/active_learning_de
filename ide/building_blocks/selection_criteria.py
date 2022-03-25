from __future__ import annotations
import imp
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np


from ide.building_blocks.test_interpolation import KNNTestInterpolator, TestInterpolator
from ide.core.configuration import Configurable
from ide.core.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List
    from typing_extensions import Self
    from ide.core.experiment_modules import ExperimentModules

class SelectionCriteria(Configurable):

    def query(self, queries) -> float:
        raise NotImplementedError

    @property
    def query_pool(self) -> QueryPool:
        raise NotImplementedError

    def __call__(self, exp_modules: ExperimentModules, *args, **kwargs) -> Self:
        return super().__call__(*args, **kwargs)

class TestSelectionCriteria(SelectionCriteria):
    test_interpolator: TestInterpolator

    def query(self, queries) -> float:
        raise NotImplementedError
    
    @property
    def query_pool(self):
        return self.test_interpolator.query_pool

    def __call__(self, exp_modules: ExperimentModules, *args, **kwargs) -> Self:
        obj = super().__call__(exp_modules, *args, **kwargs)

        obj.test_interpolator = exp_modules.test_interpolator(exp_modules)

        return obj

class PValueSelectionCriteria(TestSelectionCriteria):
    def query(self, queries):
        t, p, u = self.test_interpolator.query(queries)

        score = 1 - p
        return score
    
@dataclass
class PValueUncertaintySelectionCriteria(TestSelectionCriteria):
    explore_exploit_trade_of : float

    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = (1 - mean_p)*self.explore_exploit_trade_of + u / self.explore_exploit_trade_of

        scores = np.concatenate((score,score))

        return scores