import active_learning_de.experiments.simple_dependency_estimation as sde
from active_learning_ts.experiments.experiment_runner import ExperimentRunner
import tensorflow as tf


er = ExperimentRunner([sde], log=True)
er.run()
