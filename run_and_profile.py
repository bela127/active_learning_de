from ide.experiments.ide_lin import blueprint
from ide.core.experiment_runner import ExperimentRunner

er = ExperimentRunner([blueprint])
er.run_experiments()