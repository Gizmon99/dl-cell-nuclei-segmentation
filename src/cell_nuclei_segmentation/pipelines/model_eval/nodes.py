"""
This pipeline loads saved model and evaluaes it's performance based on dice coefficient, IoU and precision.
"""
import mmcv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector

def evaluate(images, gt, config):
    '''
    Loads the model from checkpoint and evaluates it's performance based on chosen test set and chosen metrics.
    '''
    checkpoint_file = './tutorial_exps/epoch_10.pth'
    model = init_detector(config, checkpoint_file, device='cpu')
    dice = []
    precisions = []
    iou = []
    A = np.array([[1, 2], [3, 4]])
    for i, image_name in enumerate(images):
        img = np.array(images[image_name]())
        g = np.array(gt[image_name]())
        if len(img.shape) == 2:
            img = mmcv.imconvert(img, 'gray', 'bgr')
        new_result = inference_detector(model, img)
        res = new_result.pred_instances.masks[0]
        for i in range(1, len(new_result.pred_instances.masks)):
            res = torch.logical_or(res, new_result.pred_instances.masks[i])
        res = res.numpy()*1
        precisions.append(np.sum(res == g)/(res.shape[0]*res.shape[1]))
        dice.append(2*np.sum(g == res)/(res.shape[0]*res.shape[1]+g.shape[0]*g.shape[1]))
        iou.append(np.sum(g == res)/(res.shape[0]*res.shape[1]+g.shape[0]*g.shape[1]-np.sum(g == res)))



    print("dice: ", np.average(dice))
    print("iou: ", np.average(iou))
    print("mAP: ", np.sum(precisions)/len(precisions))
    
    return 0