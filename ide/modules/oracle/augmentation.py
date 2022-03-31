from dataclasses import dataclass
from ide.core.oracle.augmentation import Augmentation

import numpy as np

@dataclass
class NoiseAugmentation(Augmentation):
    noise_ratio: float = 0.01

    rng = np.random.default_rng()

    def apply(self, data_points):

        queries, results = data_points
        
        augmented = self.rng.normal(results, self.noise_ratio)

        return queries, augmented