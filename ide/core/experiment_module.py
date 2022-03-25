from __future__ import annotations
from typing import TYPE_CHECKING

from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Self
    from ide.core.experiment_modules import ExperimentModules


class ExperimentModule(Configurable):

    def __call__(self, exp_modules: ExperimentModules, *args, **kwargs) -> Self:
        obj = super().__call__(*args, **kwargs)
        return obj