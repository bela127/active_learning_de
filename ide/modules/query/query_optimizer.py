from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from ide.core.query.query_optimizer import QueryOptimizer

if TYPE_CHECKING:
    from typing import Dict
    from ide.core.query.query_sampler import QuerySampler
    from ide.core.experiment_modules import ExperimentModules
    from typing_extensions import Self # type: ignore


@dataclass
class BayesianGPQueryOptimizer(QueryOptimizer):
    _bopt: BayesianOptimization = None
    _buf: UtilityFunction = None
    _last_queries: Dict = None

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        if not self._last_queries is None:
            queries = np.asarray(list(self._last_queries.values()))[:,None]
            scores = self.selection_criteria.query(queries)
            self._bopt.register(
                                params=self._last_queries,
                                target=float(scores[0]),
                                )

        self._last_queries = self._bopt.suggest(self._buf)

        queries = np.asarray(list(self._last_queries.values()))[:,None]

        return queries, scores #TODO Fix this, the scores are wrong currently
    
    def __call__(self, exp_modules: ExperimentModules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        query_ranges = obj.selection_criteria.query_pool.query_ranges
        obj._bopt = BayesianOptimization(
                                        f=None,
                                        pbounds={'q1': query_ranges[0], 'q2': query_ranges[0]},
                                        verbose=2,
                                        random_state=1,
                                        )
        obj._buf = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
        return obj

@dataclass
class MCQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler
    num_tries: int

    def __call__(self, exp_modules: ExperimentModules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.query_sampler = obj.query_sampler(obj.selection_criteria)
        return obj

    
class MaxMCQueryOptimizer(MCQueryOptimizer):

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        query_candidates = self.query_sampler.sample(self.num_tries)
        candidate_scores = self.selection_criteria.query(query_candidates)

        ind = np.argpartition(candidate_scores, -num_queries)[-num_queries:]
        queries = query_candidates[ind]
        scores = candidate_scores[ind]

        return queries, scores

@dataclass
class ProbWeightedMCQueryOptimizer(MCQueryOptimizer):
    _rng = np.random.default_rng()

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        query_candidates = self.query_sampler.sample(self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        
        scores_zero = np.nan_to_num(scores)
        scores_weight = np.exp(scores_zero)
        score_sum = np.sum(scores_weight)
        weight = scores_weight / score_sum

        indexes = np.arange(query_candidates.shape[0])

        if np.count_nonzero(np.isnan(weight)) > 0:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False)
        else:
            idx = self._rng.choice(a=indexes, size=num_queries, replace=False, p=weight)

        return query_candidates[idx], scores[idx]