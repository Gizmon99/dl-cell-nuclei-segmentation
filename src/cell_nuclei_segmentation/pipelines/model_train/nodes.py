from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS

@METRICS.register_module()
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)

def train_model(model, train_params, train_dataset, test_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size = train_params['batch_size'], num_workers=12, collate_fn=default_collate)
    val_dataloader = DataLoader(test_dataset, batch_size = train_params['batch_size'], num_workers=12, collate_fn=default_collate)
    runner = Runner(
        model=model,
        work_dir='logs',
        train_dataloader=train_dataloader,
        train_cfg=dict(
            by_epoch=True,   # display in epoch number instead of iterations
            max_epochs=10,
            val_begin=2,     # start validation from the 2nd epoch
            val_interval=1), # do validation every 1 epoch

        optim_wrapper=dict(
            optimizer=dict(
                type=Adam,
                lr=0.001)),

        param_scheduler=dict(
            type='MultiStepLR',
            by_epoch=True,
            milestones=[4, 8],
            gamma=0.1),

        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),

        default_hooks=dict(
            # the most commonly used hook for modifying checkpoint saving interval
            checkpoint=dict(type='CheckpointHook', interval=1)),

        launcher='none',
        env_cfg=dict(
            cudnn_benchmark=False,   # whether enable cudnn_benchmark
            backend='nccl',   # distributed communication backend
            mp_cfg=dict(mp_start_method='fork')),  # multiprocessing configs
        log_level='INFO',

        load_from=None,
        resume=False
    )

    # start training your model
    runner.train()
