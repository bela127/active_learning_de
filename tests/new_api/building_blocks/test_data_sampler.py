import numpy as np

from active_learning_de.new_api.building_blocks.data_pool import FlatDataPool
from active_learning_de.new_api.building_blocks.data_sampler import KNNDataSampler

def test_add_sample():

    dp = FlatDataPool((2,))
    dp = dp()

    sampler = KNNDataSampler(10, (2,))
    sampler = sampler()

    query = np.asarray((1,3))
    result = np.asarray((2,2))

    dp.subscrib(sampler)
    dp.add((query,result))

    queries, results = sampler.sample(query)
    assert (queries[0], results[0]) == (query,result)