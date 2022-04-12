from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Iterable, Optional

    from ide.core.data.queried_data_pool import QueriedDataPool
    from ide.core.oracle.oracle import Oracle
    from ide.core.query.query_optimizer import QueryOptimizer
    from ide.core.query.query_sampler import QuerySampler
    from ide.core.evaluator import Evaluator
    from ide.core.experiment_modules import ExperimentModules
    from ide.core.stopping_criteria import StoppingCriteria


@dataclass
class Blueprint():
    repeat: int

    stopping_criteria: StoppingCriteria

    oracle: Oracle

    queried_data_pool: QueriedDataPool

    initial_query_sampler: QuerySampler

    query_optimizer: QueryOptimizer

    experiment_modules: ExperimentModules

    evaluators: Iterable[Evaluator]

    exp_name: Optional[str]= None