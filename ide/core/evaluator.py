from ide.core.configuration import Configurable
from ide.core.experiment import Experiment

class Evaluator(Configurable):
    def register(self, experiment: Experiment):
        ...
