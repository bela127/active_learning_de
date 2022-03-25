from ide.experiments.ide_blueprint import ide_blueprint
from ide.core.experiment_runner import ExperimentRunner

er = ExperimentRunner([ide_blueprint])
er.run_experiments()
