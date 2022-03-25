from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.experiment_modules import ExperimentModules

if TYPE_CHECKING:
    from ide.building_blocks.test_interpolation import TestInterpolator

@dataclass
class DependencyExperiment(ExperimentModules):
    test_interpolator: TestInterpolator
