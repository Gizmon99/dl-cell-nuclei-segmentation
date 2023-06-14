"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
        node(func = annotate_dataset, inputs = ["train_images", "groundtruth_images"], outputs = "train_annotations"),
        node(func = annotate_dataset, inputs = ["test_images", "groundtruth_images"], outputs = "test_annotations"),
        ]
    )