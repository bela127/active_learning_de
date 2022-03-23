from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class SelectionCriteria(Configurable):
    test_interpolator: KNNTestInterpolator
    query_pool: QueryPool

    def query(self, queries) -> float:
        ...

class PValueSelectionCriteria(SelectionCriteria):
    def query(self, queries):
        t, p, u = self.test_interpolator.query(queries)
        return 1 - p

@dataclass
class PValueUncertaintySelectionCriteria(SelectionCriteria):
    explore_exploit_trade_of : float

    def query(self, queries):
        t, p, u = self.test_interpolator.query(queries)
        return (1 - p)*self.explore_exploit_trade_of + u / self.explore_exploit_trade_of