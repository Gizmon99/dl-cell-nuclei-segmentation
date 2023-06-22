from mmengine import Config
from mmengine.runner import set_random_seed

def setup_config(train_params):
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
    cfg.val_dataloader.batch_size = train_params['batch_size']

    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator.ann_file = cfg.data_root+'/'+'test/annotation_coco.json'
    cfg.test_evaluator = cfg.val_evaluator

    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1

    cfg.load_from = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    cfg.work_dir = './exps'


    cfg.train_cfg.val_interval = 10
    cfg.default_hooks.checkpoint.interval = 5

    cfg.optim_wrapper.optimizer.lr = 0.02 / 8
    cfg.default_hooks.logger.interval = 10

    cfg.train_cfg.val_interval = 10
    cfg.train_cfg.max_epochs = train_params['epochs']

    set_random_seed(0, deterministic=False)



    return cfg

