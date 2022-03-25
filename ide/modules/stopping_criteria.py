from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from ide.core.stopping_criteria import StoppingCriteria

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class LearningStepStoppingCriteria(StoppingCriteria):
    learning_steps: int

    def next(self, iteration):
        return iteration <= self.learning_steps