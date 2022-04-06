from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from ide.core.configuration import Configurable


if TYPE_CHECKING:
    from ide.core.data_pool import DataPool
    from typing_extensions import Self #type: ignore

@dataclass
class ExperimentModules(Configurable):
    data_pool: DataPool = field(init=False)

    def run(self):
        ...

    def __call__(self, data_pool: DataPool = None, **kwargs) -> Self:
        obj = super().__call__(**kwargs)
        if not data_pool is None:
            obj.data_pool = data_pool
        else:
            raise ValueError
        return obj
