<<<<<<< Updated upstream
"""
This is a boilerplate pipeline 'model_create'
generated using Kedro 0.18.6
"""
=======
import torch
import torch.nn as nn
from mmseg.models import build_segmentor


def create_model(model_cfg, train_cfg):
    """
    Returns new MMsegmentationModel model.
    """
    #config = mmcv.Config.fromfile(config_path)
    train_cfg = train_cfg

    model = build_segmentor(model_cfg)
    return model
>>>>>>> Stashed changes
