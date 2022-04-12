import numpy as np

from ide.modules.data_sampler import KDTreeKNNDataSampler
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.modules.queried_data_pool import FlatQueriedDataPool

def test_test():
    dp = FlatQueriedDataPool((1,),(2,))
    dp = dp()

    sampler = KDTreeKNNDataSampler(2)
    sampler = sampler(dp)

    dp.subscrib(sampler)

    dp.add((np.asarray((1,3)),np.asarray((3,1))))
    dp.add((np.asarray((1,2)),np.asarray((4,2))))
    dp.add((np.asarray((2,3)),np.asarray((1,1))))
    dp.add((np.asarray((2,2)),np.asarray((2,2))))

    utest = MWUTwoSampleTest(sampler)
    utest = utest()

    t,p = utest.query((np.asarray((1,2.5)),np.asarray((2,2.5))))

    assert t[0] == 16.0
    assert t[1] == 8.0

    assert p[1] == 1.0