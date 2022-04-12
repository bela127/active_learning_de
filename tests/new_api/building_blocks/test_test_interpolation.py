import numpy as np

from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.modules.data_sampler import KDTreeKNNDataSampler
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.modules.queried_data_pool import FlatQueriedDataPool

def test_uncertainty():
    dp = FlatQueriedDataPool((1,),(2,))
    dp = dp()

    sampler = KDTreeKNNDataSampler(2)
    sampler = sampler(dp)
    

    dp.add((np.asarray((1,3)),np.asarray((3,1))))
    dp.add((np.asarray((1,2)),np.asarray((4,2))))
    dp.add((np.asarray((2,3)),np.asarray((1,1))))
    dp.add((np.asarray((2,2)),np.asarray((2,2))))

    utest = MWUTwoSampleTest(sampler)
    utest = utest()

    ti = KNNTestInterpolator(data_sampler=sampler, test=utest)
    ti = ti()

    t,p,u = ti.query((np.asarray((1,2.5)),np.asarray((2,2.5))))

    assert t[0] == 4.0
    assert t[1] == 2.0

    assert p[1] == 1.0

    assert u == 0.5