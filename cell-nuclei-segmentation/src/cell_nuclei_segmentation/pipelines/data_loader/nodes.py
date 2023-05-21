import pandas as pd
from kedro.io import PartitionedDataSet
from cell_nuclei_segmentation.extras.datasets.image_dataset import ImageDataset

def create_torch_dataset(images : PartitionedDataSet, loader_params):
    """
    Returns custom torch-based dataset with images used as model inputs.
    """
    dataset = ImageDataset(images, image_size = loader_params['image_size'])

    return dataset
