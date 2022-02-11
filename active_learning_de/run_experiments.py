from active_learning_de.experiments.simple_dependency_estimation import Simple_Dependency_Estimation, dynamic_blueprints
from active_learning_ts.experiments.experiment_runner import ExperimentRunner
import tensorflow as tf


er = ExperimentRunner([Simple_Dependency_Estimation], log=True)
er.run()

er = ExperimentRunner(dynamic_blueprints)
er.run()