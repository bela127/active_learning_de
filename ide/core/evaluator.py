from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Optional
    from ide.core.experiment import Experiment

from ide.core.configuration import Configurable



class Evaluator(Configurable):
    experiment: Experiment

    def register(self, experiment: Experiment):
        self.experiment = experiment


import functools

class Evaluate:

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self._warped_func = func
        self._pre_func: Optional[Callable] = None
        self._warp_func: Optional[Callable] = None
        self._post_func: Optional[Callable] = None

    def __call__(self, *args, **kwargs):
        if not self._pre_func is None:
            self._pre_func(*args, **kwargs)
        if not self._warp_func is None:
            result = self._warp_func(self._warped_func, *args, **kwargs)
        else:
            result = self._warped_func(*args, **kwargs)
        if not self._post_func is None:
            self._post_func(result)
        return result

    
    def pre(self, methode):
        self._pre_func = methode

    def warp(self, methode):
        self._warp_func = methode

    def post(self, methode):
        self._post_func = methode
