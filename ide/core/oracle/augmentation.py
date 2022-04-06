from __future__ import annotations
from typing import TYPE_CHECKING

from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple, List, Union
    from nptyping import NDArray, Number, Shape


class Augmentation(Configurable):
    def apply(self, data_point: Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        return data_point

class NoAugmentation(Augmentation):
    ...