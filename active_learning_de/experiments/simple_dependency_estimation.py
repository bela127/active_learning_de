from typing import List
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.data_retrievement.augmentation.no_augmentation import (
    NoAugmentation,
)
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import (
    FlatMapInterpolation,
)

from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import (
    AvgRoundTimeEvaluator,
)
from active_learning_ts.instance_properties.costs.constant_instance_cost import (
    ConstantInstanceCost,
)
from active_learning_ts.instance_properties.objectives.constant_instance_objective import (
    ConstantInstanceObjective,
)
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import (
    ExactRetrievement,
)
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import (
    NoQueryOptimizer,
)
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import (
    MaximumQueryOptimizer,
)

from active_learning_ts.query_selection.query_samplers.no_query_sampler import (
    NoQuerySampler,
)
from active_learning_ts.query_selection.query_samplers.random_query_sampler import (
    RandomContinuousQuerySampler,
)
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import (
    NoSelectionCriteria,
)
from active_learning_ts.query_selection.selection_criterias.knowledge_uncertainty_selection_criteria import (
    KnowledgeUncertaintySelectionCriteria,
)
from active_learning_ts.surrogate_model.surrogate_models.no_surrogate_model import (
    NoSurrogateModel,
)
from active_learning_de.surrogate_models.pool_surrogate_model import PoolSurrogateModel


from active_learning_ts.training.training_strategies.no_training_strategy import (
    NoTrainingStrategy,
)
from active_learning_ts.training.training_strategies.direct_training_strategy import (
    DirectTrainingStrategy,
)


from distribution_data_generation.data_sources.chaotic_data_source import (
    ChaoticDataSource,
)
from distribution_data_generation.data_sources.cross_data_source import CrossDataSource
from distribution_data_generation.data_sources.cubic_data_source import CubicDataSource
from distribution_data_generation.data_sources.double_linear_data_source import (
    DoubleLinearDataSource,
)
from distribution_data_generation.data_sources.hourglass_data_source import (
    HourglassDataSource,
)
from distribution_data_generation.data_sources.hypercube_data_source import (
    HypercubeDataSource,
)
from distribution_data_generation.data_sources.inv_z_data_source import InvZDataSource
from distribution_data_generation.data_sources.linear_data_source import (
    LinearDataSource,
)
from distribution_data_generation.data_sources.linear_periodic_data_source import (
    LinearPeriodicDataSource,
)
from distribution_data_generation.data_sources.linear_then_noise_data_source import (
    LinearThenNoiseDataSource,
)
from distribution_data_generation.data_sources.multi_gausian_data_source import (
    MultiGausianDataSource,
)
from distribution_data_generation.data_sources.non_coexistence_data_source import (
    NonCoexistenceDataSource,
)
from distribution_data_generation.data_sources.power_data_source import PowerDataSource
from distribution_data_generation.data_sources.random_data_source import (
    RandomDataSource,
)
from distribution_data_generation.data_sources.sine_data_source import SineDataSource
from distribution_data_generation.data_sources.star_data_source import StarDataSource
from distribution_data_generation.data_sources.z_data_source import ZDataSource

from active_learning_de.knowledge_discovery.dependency_knowledge_task import (
    DependencyKnowledgeTask,
)

from active_learning_de.evaluator.simple_result_evaluator import SimpleResultEvaluator
from active_learning_de.evaluator.display_data_evaluator import DisplayDataEvaluator

from active_learning_ts.experiments.blueprint_element import BlueprintElement

data_sources = [
    ChaoticDataSource,
    CrossDataSource,
    CubicDataSource,
    DoubleLinearDataSource,
    HourglassDataSource,
    HypercubeDataSource,
    InvZDataSource,
    LinearDataSource,
    LinearPeriodicDataSource,
    LinearThenNoiseDataSource,
    MultiGausianDataSource,
    NonCoexistenceDataSource,
    PowerDataSource,
    RandomDataSource,
    SineDataSource,
    StarDataSource,
    ZDataSource,
]

algorithms: List = []

dynamic_blueprints = []


def create_blueprint(data_source, algorithm):
    pass


for data_source in data_sources:
    for algorithm in algorithms:
        dynamic_blueprints.append(create_blueprint(data_source, algorithm))

repeat = 1

learning_steps = 200

data_source = BlueprintElement[SineDataSource]({"dim": 1})
retrievement_strategy = BlueprintElement[ExactRetrievement]()
interpolation_strategy = BlueprintElement[FlatMapInterpolation]()

augmentation_pipeline = BlueprintElement[NoAugmentation]()

instance_level_objective = BlueprintElement[ConstantInstanceObjective]()
instance_cost = BlueprintElement[ConstantInstanceCost]()

surrogate_model = BlueprintElement[PoolSurrogateModel]()
training_strategy = BlueprintElement[DirectTrainingStrategy]()

selection_criteria = BlueprintElement[NoSelectionCriteria]()
surrogate_sampler = BlueprintElement[RandomContinuousQuerySampler]()
query_optimizer = BlueprintElement[MaximumQueryOptimizer]({"num_tries": 100})

num_knowledge_discovery_queries = 100
knowledge_discovery_sampler = BlueprintElement[RandomContinuousQuerySampler]()
knowledge_discovery_task = BlueprintElement[DependencyKnowledgeTask]()

evaluation_metrics = [
    BlueprintElement[AvgRoundTimeEvaluator](),
    BlueprintElement[SimpleResultEvaluator](),
    BlueprintElement[DisplayDataEvaluator](),
]
