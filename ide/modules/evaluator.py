from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

from ide.core.evaluator import Evaluator, Evaluate

import numpy as np
from matplotlib import pyplot as plot # type: ignore


if TYPE_CHECKING:
    from ide.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape


class PrintNewDataPointsEvaluator(Evaluator):

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.data_pool.add = Evaluate(self.experiment.data_pool.add)
        self.experiment.data_pool.add.pre(self.log_new_data_points)

    def log_new_data_points(self, data_points):
        print(data_points)


class LogNewDataPointsEvaluator(Evaluator):
    def __init__(self, logger) -> None:
        super().__init__()
        self.logger = logger

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.data_pool.add = Evaluate(self.experiment.data_pool.add)
        self.experiment.data_pool.add.pre(self.log_new_data_points)

    def log_new_data_points(self, data_points):
        # self.logger(data_points)
        ...

@dataclass
class PlotNewDataPointsEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Data"

    queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = field(init = False, default = None)
    results: NDArray[Number, Shape["query_nr, ... result_dim"]] = field(init = False, default = None)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.queried_data_pool.add = Evaluate(self.experiment.queried_data_pool.add)
        self.experiment.queried_data_pool.add.pre(self.plot_new_data_points)

        self.queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None
        self.results: NDArray[Number, Shape["query_nr, ... result_dim"]] = None
        self.iteration = 0

    def plot_new_data_points(self, data_points):

        queries, results = data_points

        queries, results = data_points

        if self.queries is None:
            self.queries = queries
            self.results = results
        else:
            self.queries = np.concatenate((self.queries, queries))
            self.results = np.concatenate((self.results, results))

        fig = plot.figure(self.fig_name)
        plot.scatter(self.queries,self.results)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

@dataclass
class PlotQueryDistEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Query distribution"

    queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = field(init = False, default = None)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.query = Evaluate(self.experiment.oracle.query)
        self.experiment.oracle.query.pre(self.plot_query_dist)

        self.queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = None

    def plot_query_dist(self, query_candidate):

        if self.queries is None:
            self.queries = query_candidate
        else:
            self.queries = np.concatenate((self.queries, query_candidate))

        fig = plot.figure(self.fig_name)
        plot.hist(self.queries)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

class PlotSampledQueriesEvaluator(Evaluator):
    interactive: bool = True
    folder: str = "fig"
    fig_name:str = "Sampled queries"

    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.query_optimizer.selection_criteria.query = Evaluate(self.experiment.query_optimizer.selection_criteria.query)
        self.experiment.query_optimizer.selection_criteria.query.pre(self.plot_queries)

    def plot_queries(self, queries):

        fig = plot.figure(self.fig_name)
        plot.scatter(queries, [0 for i in range(queries.shape[0])])
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1




