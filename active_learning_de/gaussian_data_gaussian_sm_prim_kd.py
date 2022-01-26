import active_learning_de.experiments.gaussian_data_gaussian_sm_prim_kd as blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner
import tensorflow as tf


er = ExperimentRunner([blueprint])
er.run()

test = tf.random.uniform(shape=(10, 2), minval=-5.0, maxval=5.0, seed=2)

for i in blueprint.surrogate_model.uncertainty(test):
    assert i < 1.1

for i in blueprint.knowledge_discovery_task.uncertainty(test):
    assert i < 1.1

print(blueprint.evaluation_metrics[0].get_evaluation())