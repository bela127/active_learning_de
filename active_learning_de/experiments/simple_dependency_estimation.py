from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.data_retrievement.augmentation.no_augmentation import (
    NoAugmentation,
)
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import FlatMapInterpolation

from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import (
    AvgRoundTimeEvaluator,
)
from active_learning_ts.instance_properties.costs.constant_instance_cost import (
    ConstantInstanceCost,
)
from active_learning_ts.instance_properties.objectives.constant_instance_objective import (
    ConstantInstanceObjective,
)
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import (
    ExactRetrievement,
)
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import MaximumQueryOptimizer
from active_learning_ts.query_selection.query_samplers.no_query_sampler import NoQuerySampler
from active_learning_ts.query_selection.query_samplers.random_query_sampler import (
    RandomContinuousQuerySampler,
)
from active_learning_ts.query_selection.selection_criterias.knowledge_uncertainty_selection_criteria import (
    KnowledgeUncertaintySelectionCriteria,
)
from active_learning_ts.surrogate_models.no_surrogate_model import NoSurrogateModel

from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy


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

from active_learning_de.knowledge_discovery.dependency_knowledge_task import DependencyKnowledgeTask

from active_learning_de.evaluator.simple_result_evaluator import SimpleResultEvaluator

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

algorithms = []

dynamic_blueprints = []

def create_blueprint(data_source, algorithm):
    org_init = Simple_Dependency_Estimation.__init__
    def init(self):
        org_init(self)
        self.data_source = data_source()
    Simple_Dependency_Estimation.__init__ = init
    
    return Simple_Dependency_Estimation

for data_source in data_sources:
    for algorithm in algorithms:
        dynamic_blueprints.append(create_blueprint(data_source, algorithm))



class Simple_Dependency_Estimation(Blueprint):
    repeat = 2

    def __init__(self):
        self.learning_steps = 10

        self.data_source = CrossDataSource(1)
        self.retrievement_strategy = ExactRetrievement()
        self.interpolation_strategy = FlatMapInterpolation()

        self.augmentation_pipeline = NoAugmentation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        self.surrogate_model = NoSurrogateModel()
        self.training_strategy = NoTrainingStrategy()

        self.selection_criteria = KnowledgeUncertaintySelectionCriteria()
        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = MaximumQueryOptimizer(num_tries=100)

        self.num_knowledge_discovery_queries = 1000
        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = DependencyKnowledgeTask()

        self.evaluation_metrics = [AvgRoundTimeEvaluator(), SimpleResultEvaluator()]
