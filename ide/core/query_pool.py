from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class QueryPool(Configurable):

    is_discrete: bool
    query_count: int
    query_shape: Tuple[int,...]
    query_ranges: Tuple[Tuple[Tuple[float,float],...],...]

    def elements_from_norm_pos(self, norm_pos):
        ...
    
    def elements_from_index(self, index):
        ...
    
    def all_elements(self):
        ...