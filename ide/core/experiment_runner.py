from __future__ import annotations
from typing import TYPE_CHECKING

from ide.core.experiment.blueprint import Blueprint
from ide.core.experiment_builder import Experiment

if TYPE_CHECKING:
    from typing import Tuple, List



class ExperimentRunner():
    def __init__(self) -> None:
        self.evaluators: List[EvaluationMetric] = []

    def run_experiment(self, blueprint: Blueprint):
        
        for evaluation_metric in blueprint.evaluation_metrics:
            self.evaluators.append(evaluation_metric())
        
        for exp_nr in range(blueprint.repeat):
            experiment = Experiment(blueprint, exp_nr)
        

    def run_experiments(self, blueprints: List[Blueprint]):
        for blueprint in blueprints:
            self.run_experiment(blueprint)


    def register_evaluators(self, experiment):
        for evaluator in self.evaluators:
            evaluator.register(experiment)
