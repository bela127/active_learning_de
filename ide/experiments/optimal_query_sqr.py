import numpy as np

from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import NoQueryOptimizer
from ide.modules.query.query_sampler import OptimalQuerySampler, LatinHypercubeQuerySampler, LastQuerySampler
from ide.building_blocks.selection_criteria import QueryTestNoSelectionCritera
from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.building_blocks.experiment_modules import DependencyExperiment
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
    query_optimizer=NoQueryOptimizer(
        selection_criteria=QueryTestNoSelectionCritera(),
        num_queries=4,
        query_sampler=OptimalQuerySampler(optimal_queries=(np.asarray([[0.5],[1]]),np.asarray([[1],[0.5]]),np.asarray([[0],[0.5]]),np.asarray([[0.5],[0]]))),
    ),
    experiment_modules=DependencyExperiment(
        dependency_test=DependencyTest(
            query_sampler = LastQuerySampler(),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest()
            ),
        ),
    #evaluators=[PlotQueryDistEvaluator(), PlotNewDataPointsEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],
    evaluators=[BoxPlotTestPEvaluator(folder="fig_de")],
    #evaluators=[PlotNewDataPointsEvaluator(), PlotScoresEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],

)