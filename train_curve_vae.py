import os
import sys
from argparse import ArgumentParser

# Import custom modules: model, dataset, trainer, and config utils
from src.vae.vae_curve import AutoencoderKL1D as MyModel
from src.dataset.dataset import CurveDataset as MyDataset
from src.trainer.trainer_vae import Trainer as MyTrainer
from src.utils.config import NestedDictToClass, load_config

# Arguments
# Parse command-line arguments
program_parser = ArgumentParser(description='Train curve vae model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
args, unknown = program_parser.parse_known_args()

# Load and wrap config as attribute-accessible object
cfg = load_config(args.config)
args = NestedDictToClass(cfg)

isDebug = True if sys.gettrace() else False

if isDebug:
    args.use_wandb_tracking = False
    args.batch_size = 2
    args.num_workers = 1

train_dataset = MyDataset(
    dataset_file_path = args.data.train_set_file_path,
    replication=args.data.replication,
    is_train=True
)

val_dataset = MyDataset(
    dataset_file_path = args.data.val_set_file_path,
    is_train=False,
)   

model = MyModel(
    in_channels=args.model.in_channels,
    out_channels=args.model.out_channels,
    down_block_types=args.model.down_block_types,
    up_block_types=args.model.up_block_types,
    block_out_channels=args.model.block_out_channels,
    layers_per_block=args.model.layers_per_block,
    act_fn=args.model.act_fn,
    latent_channels=args.model.latent_channels,
    norm_num_groups=args.model.norm_num_groups,
    sample_points_num=args.model.sample_points_num,
    kl_weight=args.model.kl_weight,
)

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch
num_warmup_steps = int(0.1*num_train_steps)

trainer = MyTrainer(
    model,
    dataset = train_dataset,
    val_dataset = val_dataset,
    num_train_steps = num_train_steps,
    batch_size = batch_size,
    num_workers = args.num_workers,
    num_step_per_epoch=num_step_per_epoch,
    grad_accum_every = args.grad_accum_every,
    ema_update_every = args.ema_update_every,
    learning_rate = args.lr,
    max_grad_norm = args.max_grad_norm,
    accelerator_kwargs = dict(
        cpu = False,
        step_scheduler_with_optimizer=False
    ),
    log_every_step = args.log_every_step,
    use_wandb_tracking = args.use_wandb_tracking,
    checkpoint_folder = args.model.checkpoint_folder,
    checkpoint_every_step = args.save_every_epoch * num_step_per_epoch,
    resume_training=args.resume_training,
    from_start=args.from_start,
    checkpoint_file_name=args.model.checkpoint_file_name,
    val_every_step=int(args.val_every_epoch * num_step_per_epoch),
)

# Launch training
trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg)
