from beartype import beartype
from typing import Callable, Optional
from src.trainer.trainer_base import BaseTrainer
from torch.utils.data import Dataset
from torch import nn
from einops import rearrange
from src.transport import create_transport
# trainer class
class Trainer(BaseTrainer):
    @beartype
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset, 
        *,
        val_dataset: Optional[Dataset] = None,
        batch_size = 16,
        checkpoint_folder: str = './checkpoints',
        checkpoint_every_step: int = 1000,
        checkpoint_file_name: str = 'model.pt',
        ema_update_every = 10,
        ema_decay = 0.995,
        grad_accum_every = 1,
        log_every_step: int = 10,
        learning_rate: float = 2e-4,
        mixed_precision_type = 'fp16',
        max_grad_norm: float = 1.,
        num_workers: int = 1,
        num_train_steps = 100000,
        results_folder = './results',
        resume_training = False,
        from_start = False,
        use_wandb_tracking: bool = False,
        collate_fn: Optional[Callable] = None,
        accelerator_kwargs: dict = dict(),
        val_every_step: int = 100,
        val_num_batches: int = 5,
        path_type: str = 'Linear',
        prediction: str = 'velocity',
        loss_weight = None,
        train_eps: float = 0.0,
        sample_eps: float = 0.0,
        **kwargs
    ):
        super().__init__(
            model=model, 
            dataset=dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            checkpoint_folder=checkpoint_folder,
            checkpoint_every_step=checkpoint_every_step,
            checkpoint_file_name=checkpoint_file_name,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            grad_accum_every=grad_accum_every,
            log_every_step=log_every_step,
            learning_rate=learning_rate,
            mixed_precision_type=mixed_precision_type,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            num_train_steps=num_train_steps,
            results_folder=results_folder,
            resume_training=resume_training,
            from_start=from_start,
            collate_fn=collate_fn,
            use_wandb_tracking=use_wandb_tracking,
            accelerator_kwargs=accelerator_kwargs,
            val_every_step=val_every_step,
            val_num_batches=val_num_batches,
            **kwargs
        )


        self.transport = create_transport(
            path_type,
            prediction,
            loss_weight,
            train_eps,
            sample_eps,
        )  # default: velocity; 

    def train_step(self, forward_kwargs, is_train=True):

        model_kwargs = dict(context=forward_kwargs['context'])
        x1 = forward_kwargs['zs']
    
        if is_train:
            res = self.transport.training_losses(self.model, x1, model_kwargs)
        else:
            res = self.transport.training_losses(self.ema, x1, model_kwargs)

        loss = res['loss'].mean()

        return loss, dict(loss=loss)