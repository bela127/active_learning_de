from ide.building_blocks.data_pool import FlatDataPool
from ide.building_blocks.data_sampler import KDTreeKNNDataSampler
from ide.core.oracle.oracle import Oracle
from ide.building_blocks.query_optimizer import MaxMCQueryOptimizer
from ide.building_blocks.query_sampler import RandomQuerySampler
from ide.building_blocks.selection_criteria import PValueUncertaintySelectionCriteria
from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.modules.experiment_modules import DependencyExperiment
from ide.modules.stopping_criteria import LearningStepStoppingCriteria
from ide.core.blueprint import Blueprint
from ide.core.oracle.augmentation import NoAugmentation
from ide.modules.oracle.data_source import LineDataSource


ide_blueprint = Blueprint(
    repeat=10,
    stopping_criteria= LearningStepStoppingCriteria(100),
    oracle = Oracle(
        data_source=LineDataSource((1,),(1,)),
        augmentation= NoAugmentation()
    ),
    data_pool=FlatDataPool((1,), (2,)),
    initial_query_sampler=RandomQuerySampler(num_queries=2),
    query_optimizer=MaxMCQueryOptimizer(
        selection_criteria=PValueUncertaintySelectionCriteria(explore_exploit_trade_of=0.5),
        num_queries=1,
        query_sampler=RandomQuerySampler(),
        num_tries=1000
    ),
    experiment_modules=DependencyExperiment(
        test_interpolator=KNNTestInterpolator(
            test=MWUTwoSampleTest(),
            data_sampler=KDTreeKNNDataSampler(10)
            ),
        ),
    evaluators=[],
)