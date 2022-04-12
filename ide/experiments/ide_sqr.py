from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from ide.core.oracle.oracle import Oracle
from ide.modules.query.query_optimizer import MaxMCQueryOptimizer
from ide.modules.query.query_sampler import LastQuerySampler, UniformQuerySampler, LatinHypercubeQuerySampler
from ide.building_blocks.selection_criteria import PValueSelectionCriteria, PValueUncertaintySelectionCriteria, TestScoreUncertaintySelectionCriteria, TestScoreSelectionCriteria
from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.building_blocks.experiment_modules import InterventionDependencyExperiment
from ide.modules.oracle.augmentation import NoiseAugmentation
from ide.modules.stopping_criteria import LearningStepStoppingCriteria
from ide.core.blueprint import Blueprint
from ide.modules.oracle.data_source import LineDataSource, SquareDataSource
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from ide.building_blocks.dependency_test import DependencyTest
from ide.building_blocks.multi_sample_test import KWHMultiSampleTest


blueprint = Blueprint(
    repeat=100,
    stopping_criteria= LearningStepStoppingCriteria(290),
    oracle = Oracle(
        data_source=SquareDataSource((1,),(1,)),
        augmentation= NoiseAugmentation(noise_ratio=3.0)
    ),
    queried_data_pool=FlatQueriedDataPool(),
    initial_query_sampler=LatinHypercubeQuerySampler(num_queries=10),
    query_optimizer=MaxMCQueryOptimizer(
        selection_criteria=PValueSelectionCriteria(),
        num_queries=4,
        query_sampler=LatinHypercubeQuerySampler(),
        num_tries=2000
    ),
    experiment_modules=InterventionDependencyExperiment(
        test_interpolator=KNNTestInterpolator(
            test=MWUTwoSampleTest(),
            data_sampler=KDTreeKNNDataSampler(50,sample_size_data_fraction=10)
            ),
        dependency_test=DependencyTest(
            query_sampler = LatinHypercubeQuerySampler(num_queries=20),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest()
            ),
        ),
    #evaluators=[PlotQueryDistEvaluator(), PlotNewDataPointsEvaluator(), PlotScoresEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],
    evaluators=[BoxPlotTestPEvaluator(folder="fig_ide")],
    #evaluators=[ PlotQueriesEvaluator()],

)