from mmengine.runner import Runner
from mmdet.utils import AvoidCUDAOOM
import torch
import wandb



def train_model(model_config):
    
    wandb.login()

    torch.cuda.set_per_process_memory_fraction(0.95, 0)
    runner = Runner.from_cfg(model_config)
    runner.train()

    return 0
    
