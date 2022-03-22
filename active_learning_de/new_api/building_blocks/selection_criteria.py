from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from active_learning_de.new_api.building_blocks.data_sampler import DataSampler
from active_learning_de.new_api.core.configuration import Configurable

if TYPE_CHECKING:
    from active_learning_de.new_api.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class SelectionCriteria(Configurable):
    def query(self, queries):
        ...

class PValueSelectionCriteria(SelectionCriteria):
    def query(self, queries):
        ...

class PValueUncertaintySelectionCriteria(SelectionCriteria):
    def query(self, queries):
        ...