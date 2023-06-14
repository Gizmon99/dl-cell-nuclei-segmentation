"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.18.6
"""
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer

def evaluate(images, config):
    

    for image_name in images:
        img = np.array(images[image_name]())
        break
    img = mmcv.imread('/home/pslowiq/programs/dl-cell-nuclei-segmentation/data/01_raw/dataset/train/Ganglioneuroblastoma_0.tif')
    checkpoint_file = './tutorial_exps/epoch_10.pth'
    model = init_detector(config, checkpoint_file, device='cuda')
    new_result = inference_detector(model, img)

    print(new_result)
    return 0