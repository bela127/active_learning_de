from __future__ import annotations
from abc import abstractmethod, abstractproperty
import imp
from typing import TYPE_CHECKING
from ide.core.data.data_pool import DataPool


from ide.core.experiment_module import ExperimentModule
from ide.core.queryable import Queryable

if TYPE_CHECKING:
    from typing import Tuple
    from nptyping import NDArray, Shape, Number
    



class SelectionCriteria(ExperimentModule, Queryable):

    @abstractmethod
    def query(self, queries: NDArray[Number,  Shape["query_nr, ... query_shape"]]) -> Tuple[NDArray[Number,  Shape["query_nr, ... query_shape"]], NDArray[Number,  Shape["query_nr, 1 score"]]]:
        raise NotImplementedError

    @property
    def data_pool(self):
        return DataPool(self.query_pool, (self.query_pool.query_shape[0], 1))

class NoSelectionCriteria(SelectionCriteria):

    def query(self, queries):
        raise NameError("query(queries), should never be called on NoSelectionCriteria")
