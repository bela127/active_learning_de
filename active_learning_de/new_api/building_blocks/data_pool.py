from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

from active_learning_de.new_api.core.configuration import Configurable
from active_learning_de.new_api.building_blocks.data_subscriber import DataSubscriber

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class DataPool(Configurable):

    shape: Tuple
    _subscriber: List[DataSubscriber] = field(default_factory=list, init=False)

    def subscrib(self, subscriber):
        self._subscriber.append(subscriber)


    def add(self, data_point):
        for subscriber in self._subscriber:
            subscriber.update(data_point)

@dataclass
class FlatDataPool(DataPool):
    """
    implements a pool of already labeled data
    """

    def __init__(self, shape):
        super().__init__(shape)
        self.queries: List = []
        self.results: List = []

    def add(self, data_point):
        super().add(data_point)
        query, result = data_point
        self.queries.append(query)
        self.results.append(result)
