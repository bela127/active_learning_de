from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Tuple, List, Union
    from nptyping import NDArray, Number, Shape


@dataclass
class QueryPool():

    query_count: Union[int, None]
    query_shape: Tuple[int, ...]
    query_ranges: NDArray[Number, Shape["... query_dims,[xi_min, xi_max]"]]

    def elements_from_norm_pos(self, norm_pos: NDArray[Number, Shape["query_nr, ... query_dims"]]) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        elements = self.query_ranges[..., 0] + (self.query_ranges[..., 1] - self.query_ranges[..., 0]) * norm_pos
        return elements
    
    def elements_from_index(self, index):
        raise NotImplementedError
    
    def all_elements(self):
        raise NotImplementedError