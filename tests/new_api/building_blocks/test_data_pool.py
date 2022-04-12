import numpy as np

from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.core.data_subscriber import DataSubscriber


def test_data_pool_add():
    dp = FlatQueriedDataPool((1,),(2,))
    dp = dp()
    assert dp.result_shape == (2,)

    query = np.asarray((1,))
    result = np.asarray((2,2))

    dp.add((query,result))
    assert np.all(dp.queries[0] == query)
    assert np.all(dp.results[0] == result)


def test_data_pool_subscribe():
    dp = FlatQueriedDataPool((1,),(2,))
    dp = dp()

    queries = np.asarray(((1,),))
    results = np.asarray(((2,2),))

    class Test_Subscriber(DataSubscriber):
        def update(self, data_point):
            queries_u, results_u = data_point
            assert np.all(queries_u == queries)
            assert np.all(results_u == results)
            
    sub = Test_Subscriber()
    sub = sub()
    dp.subscrib(sub)

    assert dp._subscriber[0] == sub

    dp.add((queries,results))