"""
This is a boilerplate pipeline 'mask_rnn_setup'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *
        

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func = setup_config, inputs = None, outputs = 'mask_rcnn_config')
    ])
