from __future__ import annotations
from typing import TYPE_CHECKING

from ide.building_blocks.experiment_modules import DependencyExperiment
from dataclasses import dataclass, field
from ide.core.evaluator import Evaluator, Evaluate

import numpy as np
from matplotlib import pyplot as plot # type: ignore


if TYPE_CHECKING:
    from typing import List
    from ide.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape
    from ide.building_blocks.dependency_test import DependencyTest


@dataclass
class PlotQueriesEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Query distribution 2d"

    queries: NDArray[Number, Shape["2, query_nr, ... query_dim"]] = field(init=False, default = None)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.query = Evaluate(self.experiment.oracle.query)
        self.experiment.oracle.query.pre(self.plot_queries)

        self.queries: NDArray[Number, Shape["2, query_nr, ... query_dim"]] = None


    def plot_queries(self, queries):

        size = queries.shape[0] // 2
        queries = np.reshape(queries, (2, size,-1))

        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries), axis=1)


        heatmap, xedges, yedges = np.histogram2d(self.queries[0,:,0], self.queries[1,:,0], bins=10)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        
        fig = plot.figure(self.fig_name)
        plot.imshow(heatmap.T, extent=extent, origin='lower')
        plot.title(self.fig_name)
        plot.colorbar()
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

@dataclass
class PlotScoresEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Scores 2d"

    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.query_optimizer.selection_criteria.query = Evaluate(self.experiment.query_optimizer.selection_criteria.query)
        self.experiment.query_optimizer.selection_criteria.query.warp(self.plot_scores)

    def plot_scores(self, func, queries):

        scores = func(queries)

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))
        test_scores = scores[:size]

        fig = plot.figure(self.fig_name)
        plot.scatter(test_queries[:,0,0], test_queries[:,1,0], c=test_scores)
        plot.title(self.fig_name)
        plot.colorbar()
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

        return scores



@dataclass
class PlotTestPEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "P-value"

    ps: List[float] = field(init=False, default_factory=list)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.experiment_modules, DependencyExperiment):
            self.experiment.experiment_modules.dependency_test.test = Evaluate(self.experiment.experiment_modules.dependency_test.test)
            self.experiment.experiment_modules.dependency_test.test.post(self.plot_test_result)
        else:
            raise ValueError

        self.ps = []


    def plot_test_result(self, result):
        t,p = result

        self.ps.append(p[0])

        fig = plot.figure(self.fig_name)
        plot.plot([i for i in range(len(self.ps))], self.ps)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1


@dataclass
class BoxPlotTestPEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Boxplot p-value"

    ps: List[float] = field(init=False, default_factory=list)
    pss: List[float] = field(init=False, default_factory=list)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.experiment_modules, DependencyExperiment):
            self.experiment.experiment_modules.dependency_test.test = Evaluate(self.experiment.experiment_modules.dependency_test.test)
            self.experiment.experiment_modules.dependency_test.test.post(self.save_test_result)
        else:
            raise ValueError

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.plot_test_results)

        self.ps = []
    
    def save_test_result(self, result):
        t,p = result

        self.ps.append(p[0])

    def plot_test_results(self, _):

        self.pss.append(self.ps)

        data = np.asarray(self.pss)
        positions = np.arange(data.shape[1]) + 1

        fig = plot.figure(self.fig_name)

        plot.boxplot(data, positions=positions, meanline=False, showmeans=False, showfliers=False)
        means = np.mean(data, axis=0)
        plot.plot(positions, means)
        plot.xticks(np.arange(data.shape[1], step=10),np.arange(data.shape[1], step=10))
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png',dpi=500)
            plot.clf()

        self.iteration += 1


