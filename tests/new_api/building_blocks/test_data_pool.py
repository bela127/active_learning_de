import numpy as np

from active_learning_de.new_api.building_blocks.data_pool import FlatDataPool
from active_learning_de.new_api.building_blocks.data_subscriber import DataSubscriber


def test_data_pool_add():
    dp = FlatDataPool((2,))
    dp = dp()
    assert dp.shape == (2,)

    query = np.asarray((1,))
    result = np.asarray((2,2))

    dp.add((query,result))
    assert np.all(dp.queries[0] == query)
    assert np.all(dp.results[0] == result)


def test_data_pool_subscribe():
    dp = FlatDataPool((2,))
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