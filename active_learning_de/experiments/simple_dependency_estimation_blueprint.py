from dataclasses import dataclass
from typing import Iterable


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
class SimpleDependencyBlueprint(DynamicBlueprint):
    knowledge_discovery_task: BlueprintElement[KnowledgeDiscoveryTask] = BlueprintElement[DependencyKnowledgeTask]()
    evaluation_metrics: Iterable[BlueprintElement[EvaluationMetric]] = (
    BlueprintElement[SimpleResultEvaluator](),
    BlueprintElement[DisplayDataEvaluator](),
    )

