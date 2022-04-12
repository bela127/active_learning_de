from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.experiment_modules import ExperimentModules

if TYPE_CHECKING:
    from ide.building_blocks.test_interpolation import TestInterpolator
    from ide.building_blocks.dependency_test import DependencyTest
    from ide.core.data.queried_data_pool import QueriedDataPool
    from ide.core.data.data_pool import DataPool

    from typing_extensions import Self #type: ignore


@dataclass
class DependencyExperiment(ExperimentModules):
    dependency_test: DependencyTest

    def run(self):
        t,p = self.dependency_test.test()

    def __call__(self, queried_data_pool: QueriedDataPool = None, oracle_data_pool: DataPool = None, **kwargs) -> Self:
        obj = super().__call__(queried_data_pool, oracle_data_pool, **kwargs)
        obj.dependency_test = obj.dependency_test(obj)
        return obj

@dataclass
class InterventionDependencyExperiment(DependencyExperiment):
    test_interpolator: TestInterpolator

    def run(self):
        t,p = self.dependency_test.test()

    def __call__(self, queried_data_pool: QueriedDataPool = None, oracle_data_pool: DataPool = None, **kwargs) -> Self:
        obj = super().__call__(queried_data_pool, oracle_data_pool, **kwargs)
        obj.test_interpolator = obj.test_interpolator(obj)
        return obj
