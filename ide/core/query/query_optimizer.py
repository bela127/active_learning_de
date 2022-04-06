from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.experiment_module import ExperimentModule

if TYPE_CHECKING:
    from typing_extensions import Self

    from ide.core.query.selection_criteria import SelectionCriteria
    from ide.core.experiment_modules import ExperimentModules
    from ide.core.query.query_sampler import QuerySampler

    from nptyping import NDArray, Number, Shape
    
@dataclass
class QueryOptimizer(ExperimentModule):
    selection_criteria: SelectionCriteria
    num_queries: int

    def select(self, num_queries = None) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        if num_queries is None: num_queries = self.num_queries
        raise NotImplementedError

    def __call__(self, exp_modules: ExperimentModules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.selection_criteria = obj.selection_criteria(exp_modules)
        return obj

@dataclass
class NoQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        queries = self.query_sampler.sample(self.selection_criteria.query_pool, self.num_queries)

        return queries