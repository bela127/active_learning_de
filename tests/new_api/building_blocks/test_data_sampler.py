import numpy as np

from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.data_sampler import KDTreeKNNDataSampler

def test_add_sample():

    dp = FlatQueriedDataPool((1,),(2,))
    dp = dp()

    sampler = KDTreeKNNDataSampler(10, (2,))
    sampler = sampler(dp)

    query = np.asarray([(1,3)])
    result = np.asarray([(2,2)])

    dp.add((query,result))

    queries, results = sampler.sample(query)
    assert (queries[0], results[0]) == (query,result)