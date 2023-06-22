"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func = evaluate, inputs = ['params:checkpoint_filepath','train_images', 'groundtruth_images', 'mask_rcnn_config'], outputs = ['dice', 'iou', 'map'])
    ])
