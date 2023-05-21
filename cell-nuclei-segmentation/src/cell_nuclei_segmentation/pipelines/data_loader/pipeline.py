"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
        node(func = create_torch_dataset, inputs = ["train_pre", "params:loader_params"], outputs = "train_dataset"),
        node(func = create_torch_dataset, inputs = ["test_pre", "params:loader_params"], outputs = "test_dataset")
        ]
    )