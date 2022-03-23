from dataclasses import dataclass
from ide.building_blocks.data_pool import DataPool

from ide.building_blocks.oracle import Oracle
from ide.building_blocks.query_optimizer import QueryOptimizer
from ide.building_blocks.query_sampler import QuerySampler
from ide.building_blocks.stopping_criteria import StoppingCriteria


@dataclass
class Blueprint():
    repeat: int

    stopping_criteria: StoppingCriteria

    oracle: Oracle

    data_pool: DataPool

    initial_query_sampler: QuerySampler

    query_optimizer: QueryOptimizer

    #evaluation_metrics: Iterable[EvaluationMetric]