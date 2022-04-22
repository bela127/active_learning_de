from __future__ import annotations
import imp
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from ide.core.data.data_pool import DataPool

from ide.core.query.selection_criteria import SelectionCriteria, NoSelectionCriteria
from ide.building_blocks.experiment_modules import DependencyExperiment, InterventionDependencyExperiment


if TYPE_CHECKING:
    from typing import Optional
    from typing_extensions import Self #type: ignore
    from ide.building_blocks.test_interpolation import TestInterpolator
    from ide.building_blocks.dependency_test import DependencyTest
    from ide.core.oracle.data_source import DataSource

@dataclass
class QueryTestNoSelectionCritera(NoSelectionCriteria):
    dependency_test: Optional[DependencyTest] = field(init=False, default=None)
    
    @property
    def query_pool(self):
        return self.dependency_test.data_sampler.query_pool

    
    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)

        if isinstance(exp_modules, DependencyExperiment):
            obj.dependency_test = exp_modules.dependency_test
        else:
            raise ValueError

        return obj

@dataclass
class TestSelectionCriteria(SelectionCriteria):
    test_interpolator: Optional[TestInterpolator] = field(init=False, default=None)
    
    @property
    def query_pool(self):
        return self.test_interpolator.query_pool

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)

        if isinstance(exp_modules, InterventionDependencyExperiment):
            obj.test_interpolator = exp_modules.test_interpolator
        else:
            raise ValueError()

        return obj

class PValueSelectionCriteria(TestSelectionCriteria):
    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = 1 - mean_p

        scores = np.repeat(score,2)

        return scores
    
@dataclass
class PValueUncertaintySelectionCriteria(TestSelectionCriteria):
    explore_exploit_trade_of : float = 0.5

    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = u * self.explore_exploit_trade_of + (1 - mean_p) * (1 - self.explore_exploit_trade_of)

        scores = np.repeat(score,2)

        return scores

@dataclass
class PValueDensitySelectionCriteria(TestSelectionCriteria):

    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = (1 - mean_p) * u

        scores = np.repeat(score,2)

        return scores


@dataclass
class TestScoreUncertaintySelectionCriteria(TestSelectionCriteria):
    explore_exploit_trade_of : float = 0.5

    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_t = np.mean(t, axis=1)

        score = mean_t*self.explore_exploit_trade_of + u * (1 - self.explore_exploit_trade_of)

        scores = np.repeat(score,2)

        return scores

@dataclass
class TestScoreSelectionCriteria(TestSelectionCriteria):

    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_t = np.mean(t, axis=1)

        score = mean_t

        scores = np.repeat(score,2)

        return scores

@dataclass
class OptimalSelectionCriteria(TestSelectionCriteria):
    data_source: DataSource

    def query(self, queries):

        queries, results = self.data_source.query(queries)

        size = results.shape[0] // 2
        test_results = np.reshape(results, (size,2,-1))

        score = np.abs(test_results[:,0,:] - test_results[:,1,:])

        mean_score = np.mean(score, axis=1)

        scores = np.repeat(mean_score,2)

        return scores

    def __call__(self, exp_modules=None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.data_source = obj.data_source()
        return obj
