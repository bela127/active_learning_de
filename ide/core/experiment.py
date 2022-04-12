from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ide.core.blueprint import Blueprint
    from nptyping import NDArray, Number, Shape


class Experiment():
    exp_nr: int

    def __init__(self, bp: Blueprint, exp_nr: int) -> None:
        self.exp_nr = exp_nr

        self.oracle = bp.oracle()

        self.queried_data_pool = bp.queried_data_pool(self.oracle.query_pool, self.oracle.data_pool)
        self.experiment_modules = bp.experiment_modules(self.queried_data_pool, self.oracle.data_pool)

        self.initial_query_sampler = bp.initial_query_sampler()
        self.query_optimizer = bp.query_optimizer(self.experiment_modules)
        self.stopping_criteria = bp.stopping_criteria()



    def run(self):
        iteration = 0
        queries = self.initial_query_sampler.sample(self.oracle.query_pool)
        while self.stopping_criteria.next(iteration):
            queries = self.loop(iteration, queries)
            self.experiment_modules.run()
            iteration += 1

    def loop(self, iteration: int, queries: NDArray[Number, Shape["query_nr, ... query_dims"]]) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        data_points = self.oracle.query(queries)
        self.queried_data_pool.add(data_points)
        queries = self.query_optimizer.select()
        return queries
            