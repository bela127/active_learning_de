from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ide.building_blocks.data_pool import DataPool

@dataclass
class ExperimentModules():
    data_pool: DataPool = field(init=False)
