from typing_extensions import Self
from ide.building_blocks.data_pool import DataPool
from ide.building_blocks.data_sampler import DataSampler
from ide.core.configuration import Configurable

class Oracle(Configurable):
    """
    Uses the given retrievement strategy in order to retrieve data from the given data source
    """
    data_source: DataPool
    data_sampler: DataSampler
    interpolation_strategy: InterpolationStrategy
    augmentation_pipeline: PipelineElement



    def query(self, query_candidates):

        data_points = self.data_sampler.sample(query_candidates)
        data_points = self.interpolation_strategy.interpolate(data_points)
        augmented_data_points = self.augmentation_pipeline.apply(data_points)

        return augmented_data_points

    @property
    def query_pool(self):
        self.data_sampler.query_pool

    def __call__(self, *args, **kwargs) -> Self:
        obj = super().__call__(*args, **kwargs)

        obj.data_source = obj.data_source()
        obj.data_sampler = obj.data_sampler(obj.data_source)
        obj.interpolation_strategy = obj.interpolation_strategy()
        obj.augmentation_pipeline = obj.augmentation_pipeline()

        return obj



