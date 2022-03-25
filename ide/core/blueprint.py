from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Iterable

    from ide.building_blocks.data_pool import DataPool
    from ide.core.oracle.oracle import Oracle
    from ide.building_blocks.query_optimizer import QueryOptimizer
    from ide.building_blocks.query_sampler import QuerySampler
    from ide.core.evaluator import Evaluator
    from ide.core.experiment_modules import ExperimentModules
    from ide.core.stopping_criteria import StoppingCriteria


@dataclass
class Blueprint():
    repeat: int

    stopping_criteria: StoppingCriteria

    oracle: Oracle

    data_pool: DataPool

    initial_query_sampler: QuerySampler

    query_optimizer: QueryOptimizer

    experiment_modules: ExperimentModules

    evaluators: Iterable[Evaluator]