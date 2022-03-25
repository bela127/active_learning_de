from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ide.core.blueprint import Blueprint


class Experiment():
    exp_nr: int

    def __init__(self, bp: Blueprint, exp_nr: int) -> None:
        self.exp_nr = exp_nr

        self.experiment_modules = bp.experiment_modules #TODO

        self.oracle = bp.oracle()
        self.data_pool = bp.data_pool(self.oracle.query_pool)

        self.experiment_modules.data_pool = self.data_pool #TODO

        self.initial_query_sampler = bp.initial_query_sampler()
        self.query_optimizer = bp.query_optimizer(self.experiment_modules)
        self.stopping_criteria = bp.stopping_criteria()



    def run(self):
        iteration = 0
        queries = self.initial_query_sampler.sample(self.oracle.query_pool)
        while self.stopping_criteria.next(iteration):
            data_points = self.oracle.query(queries)
            self.data_pool.add(data_points)

            queries = self.query_optimizer.select()
            
            iteration += 1