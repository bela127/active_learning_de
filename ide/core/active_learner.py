from dataclasses import dataclass

from ide.building_blocks.oracle import Oracle
from ide.building_blocks.data_pool import DataPool
from ide.building_blocks.query_sampler import QuerySampler
from ide.building_blocks.query_optimizer import QueryOptimizer
from ide.building_blocks.stopping_criteria import StoppingCriteria

@dataclass
class ActiveLearner():
    oracle: Oracle
    data_pool: DataPool
    initial_query_sampler: QuerySampler
    query_optimizer: QueryOptimizer
    stopping_criteria: StoppingCriteria


    def loop(self):
        iteration = 0
        queries = self.initial_query_sampler.sample(self.oracle.pool)
        while self.stopping_criteria.next(iteration):
            data_points = self.oracle.query(queries)
            self.data_pool.add(data_points)

            queries = self.query_optimizer.select()
            
            iteration += 1
