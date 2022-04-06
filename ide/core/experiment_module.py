from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from ide.core.experiment_modules import ExperimentModules

@dataclass
class ExperimentModule(Configurable):
    exp_modules: ExperimentModules = field(init=False)

    def __call__(self, exp_modules: ExperimentModules = None, **kwargs) -> Self:
        obj = super().__call__( **kwargs)
        obj.exp_modules = exp_modules
        return obj