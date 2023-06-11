"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func = train_model, inputs = ['mm_model', 'params:training_params', 'dataset', 'dataset'], outputs = 'trained_model')
    ])
