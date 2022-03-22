import numpy as np

from active_learning_de.new_api.building_blocks.data_sampler import KNNDataSampler
from active_learning_de.new_api.building_blocks.two_sample_test import MWUTwoSampleTest
from active_learning_de.new_api.building_blocks.data_pool import FlatDataPool

def test_test():
    dp = FlatDataPool((2,))
    dp = dp()

    sampler = KNNDataSampler(2, (2,))
    sampler = sampler()

    dp.subscrib(sampler)

    dp.add((np.asarray((1,3)),np.asarray((3,1))))
    dp.add((np.asarray((1,2)),np.asarray((4,2))))
    dp.add((np.asarray((2,3)),np.asarray((1,1))))
    dp.add((np.asarray((2,2)),np.asarray((2,2))))

    utest = MWUTwoSampleTest(sampler)
    utest = utest()

    t,p = utest.query((np.asarray((1,2.5)),np.asarray((2,2.5))))

    assert t[0] == 4.0
    assert t[1] == 2.0

    assert p[1] == 1.0