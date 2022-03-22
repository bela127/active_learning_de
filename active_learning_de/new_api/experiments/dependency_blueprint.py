from active_learning_de.blueprint.dynamic_blueprint import DynamicBlueprint
from active_learning_ts.experiments.blueprint_element import BlueprintElement

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric

from active_learning_de.evaluator.simple_result_evaluator import SimpleResultEvaluator
from active_learning_de.evaluator.display_data_evaluator import DisplayDataEvaluator
from active_learning_de.knowledge_discovery.dependency_knowledge_task import (
    DependencyKnowledgeTask,
)

@dataclass(frozen=True)
class DependencyBlueprint(AlBlueprint):
    learning_steps: int = 100
    repeat: int = 20

    oracle_pl: OraclePipeline = OraclePipeline(
        data_source = TestDataSource(),
        augmentation_pl = NoAugmentationPipeline(),
        retrievement = ExactRetrievement(),
        interpolation = FlatMapInterpolation(),
    )

    surrogate_pl: SurrogatePipeline = SurrogatePipeline(
        surrogate = PoolSurrogate(),
        training = DirectPoolTraining(),
    )

    post_oracle_pl: PostOraclePipeline = NoPostOraclePipeline()

    querry_pl: QueryPipeline = QueryPipeline(
        query_optimizer = MonteCarloMaximaOptimizer(
            utility_measure = RandomUtility(),
            query_syntesiser = RandomQueries(query_count = 1),
        )
    )

    knowledge_pl: KnowledgePipeline = KnowledgePipeline(
        query_syntesiser = AllQueries(),
        knowledge_task = DependencyKnowledgeTask(),
    )

    evaluation_metrics: Iterable[EvaluationMetric] = []