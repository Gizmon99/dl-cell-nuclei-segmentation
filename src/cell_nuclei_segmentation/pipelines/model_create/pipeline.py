"""
This is a boilerplate pipeline 'model_create'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func = create_model, inputs = ['params:model', 'params:train_cfg'], outputs = 'mm_model')
    ])
