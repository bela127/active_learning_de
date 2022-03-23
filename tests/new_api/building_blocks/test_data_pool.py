import numpy as np

from ide.building_blocks.data_pool import FlatDataPool
from ide.building_blocks.data_subscriber import DataSubscriber


def test_data_pool_add():
    dp = FlatDataPool((1,),(2,))
    dp = dp()
    assert dp.result_shape == (2,)

    query = np.asarray((1,))
    result = np.asarray((2,2))

    dp.add((query,result))
    assert np.all(dp.queries[0] == query)
    assert np.all(dp.results[0] == result)


def test_data_pool_subscribe():
    dp = FlatDataPool((1,),(2,))
    dp = dp()

    query = np.asarray((1,))
    result = np.asarray((2,2))

    class Test_Subscriber(DataSubscriber):
        def update(self, data_point):
            queries, results = data_point
            assert np.all(queries == query)
            assert np.all(results == result)
            
    sub = Test_Subscriber()
    sub = sub()
    dp.subscrib(sub)

    assert dp._subscriber[0] == sub

    dp.add((query,result))