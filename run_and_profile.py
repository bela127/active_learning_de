from ide.experiments.ide_lin import blueprint
from ide.core.experiment_runner import ExperimentRunner

if __name__ == '__main__': 
    er = ExperimentRunner([blueprint])
    er.run_experiments()