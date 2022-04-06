from ide.modules.data_pool import FlatDataPool
from ide.modules.data_sampler import KDTreeKNNDataSampler
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import NoQueryOptimizer
from ide.modules.query.query_sampler import RandomQuerySampler, LatinHypercubeSampler
from ide.building_blocks.selection_criteria import QueryTestNoSelectionCritera
from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.building_blocks.experiment_modules import DependencyExperiment
from ide.modules.oracle.augmentation import NoiseAugmentation
from ide.modules.stopping_criteria import LearningStepStoppingCriteria
from ide.core.blueprint import Blueprint
from ide.modules.oracle.data_source import LineDataSource, SquareDataSource
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator
from ide.building_blocks.dependency_test import DependencyTest
from ide.building_blocks.multi_sample_test import KWHMultiSampleTest


de_blueprint = Blueprint(
    repeat=1,
    stopping_criteria= LearningStepStoppingCriteria(80),
    oracle = Oracle(
        data_source=LineDataSource((1,),(1,)),
        augmentation= NoiseAugmentation(noise_ratio=1.0)
    ),
    data_pool=FlatDataPool((1,), (1,)),
    initial_query_sampler=LatinHypercubeSampler(num_queries=20),
    query_optimizer=NoQueryOptimizer(
        selection_criteria=QueryTestNoSelectionCritera(),
        num_queries=4,
        query_sampler=LatinHypercubeSampler(),
    ),
    experiment_modules=DependencyExperiment(
        dependency_test=DependencyTest(
            LatinHypercubeSampler(num_queries=100),
            data_sampler=KDTreeKNNDataSampler(50),
            multi_sample_test=KWHMultiSampleTest()
            ),
        ),
    evaluators=[PlotQueryDistEvaluator(), PlotNewDataPointsEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator()],
)