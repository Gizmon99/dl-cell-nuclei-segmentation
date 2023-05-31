import pandas as pd
from kedro.io import PartitionedDataSet
from cell_nuclei_segmentation.extras.datasets.image_dataset import ImageDataset

def create_torch_dataset(raw_images : PartitionedDataSet, target_masks : PartitionedDataSet):
    """
    Returns custom torch-based dataset with images used as model inputs.
    """
    dataset = ImageDataset(raw_images, target_masks)

    return dataset
