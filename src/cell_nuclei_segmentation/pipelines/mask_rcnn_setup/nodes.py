from mmengine import Config
from mmengine.runner import set_random_seed

def setup_config():
    '''
    Sets us model configurations, such as where to derive data from and model structure.
    '''
    cfg = Config.fromfile('./configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py')
    cfg.metainfo = {
        'classes': ('nuclei', ),
        'palette': [
            (220, 20, 60),
        ]
    }
    #Fixes CUDA OOM Error
    cfg.model.backbone.with_cp = True

    # Modify dataset type and path
    cfg.data_root = './data/01_raw/dataset'

    cfg.train_dataloader.dataset.ann_file = 'train/annotation_coco.json'
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'train/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    cfg.train_dataloader.batch_size = 1

    cfg.val_dataloader.dataset.ann_file = 'test/annotation_coco.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = 'test/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.batch_size = 1

    cfg.test_dataloader = cfg.val_dataloader

    # Modify metric config
    cfg.val_evaluator.ann_file = cfg.data_root+'/'+'test/annotation_coco.json'
    cfg.test_evaluator = cfg.val_evaluator

    # Modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'


    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 10
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 10

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optim_wrapper.optimizer.lr = 0.02 / 8
    cfg.default_hooks.logger.interval = 10

    #cfg.runner.max_epochs = 50


    # Setup Wandb
    #print(cfg.keys())



    # Set seed thus the results are more reproducible
    # cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # We can also use tensorboard to log the training process
    # cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})


    return cfg

