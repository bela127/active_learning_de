from ide.core.experiment.blueprint import Blueprint
from ide.core.active_learner import ActiveLearner

class Experiment():
    active_learner: ActiveLearner
    exp_nr: int

    def __init__(self, bp: Blueprint, exp_nr: int) -> None:
        self.exp_nr = exp_nr

        self.active_learner = ActiveLearner(
            bp.oracle(),
            bp.data_pool(),
            bp.initial_query_sampler(),
            bp.query_optimizer(),
            bp.stopping_criteria(),
        )

    def run(self):
        self.active_learner.loop()