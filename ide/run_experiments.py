from ide.experiments.simple_dependency_estimation import dynamic_blueprints
from ide.experiments.simple_dependency_estimation import test_blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner


er = ExperimentRunner(dynamic_blueprints, log=False)
er.run()
