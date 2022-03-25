from ide.core.configuration import Configurable


class Augmentation(Configurable):
    def apply(self, data_point):
        return data_point

class NoAugmentation(Augmentation):
    ...