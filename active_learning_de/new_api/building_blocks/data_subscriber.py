from __future__ import annotations
from typing import TYPE_CHECKING

from active_learning_de.new_api.core.configuration import Configurable

if TYPE_CHECKING:
    ...

class DataSubscriber(Configurable):

    def update(self, data_point):
        ...