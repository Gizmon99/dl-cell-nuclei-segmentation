import torch
import torch.nn as nn
from mmseg.models import build_segmentor
from mmcv import Config
import pytorch_lightning as pl


class MMsegmentationModel(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg):
        super().__init__()
        
        # config = Config.fromfile(config_path)
        self.train_cfg = train_cfg

        self.model = build_segmentor(model_cfg)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = self.criterion(outputs, targets)
        
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = self.criterion(outputs, targets)
        
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.train_cfg.lr, momentum=self.train_cfg.momentum, weight_decay=self.train_cfg.weight_decay)
        return optimizer


def create_model(model, cfg):
    """
    Returns new MMsegmentationModel model.
    """

    return MMsegmentationModel(model, cfg)
