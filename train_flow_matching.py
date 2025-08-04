import sys

from argparse import ArgumentParser
from src.utils.config import NestedDictToClass, load_config

from src.flow.flow import DiT as MyModel
from src.dataset.dataset import LatentDataset
from src.trainer.trainer_flow import Trainer as MyTrainer

# Arguments
parser = ArgumentParser(description='Train flow matching model.')
parser.add_argument('--config', type=str, default='', help='Path to config file.')

program_args = parser.parse_args()

cfg = load_config(program_args.config)
args = NestedDictToClass(cfg)

isDebug = True if sys.gettrace() else False

if isDebug:
    args.use_wandb_tracking = False
    args.data.replication = 1
    args.batch_size = 2
    args.num_workers = 1
    args.num_gpus = 1

train_dataset = LatentDataset(
    dataset_file_path = args.data.train_set_file_path,
    replication=args.data.replication,
    is_train=True,
    condition_on_points=args.model.condition_on_points,
    use_partial_pc=args.data.use_partial_pc if hasattr(args.data, 'use_partial_pc') else False,
    condition_on_img=args.model.condition_on_img if hasattr(args.model, 'condition_on_img') else False,
)

val_dataset = LatentDataset(
    dataset_file_path = args.data.val_set_file_path,
    is_train=False,
    condition_on_points=args.model.condition_on_points,
    use_partial_pc=args.data.use_partial_pc if hasattr(args.data, 'use_partial_pc') else False,
    condition_on_img=args.model.condition_on_img if hasattr(args.model, 'condition_on_img') else False,
)   
    
model = MyModel(
    in_channels=args.model.in_channels,
    latent_num=args.model.latent_num,
    hidden_size=args.model.hidden_size,
    num_heads=args.model.num_heads,
    depth=args.model.depth,
    mlp_ratio=args.model.mlp_ratio,
    condition_on_points=args.model.condition_on_points,
    condition_on_img=args.model.condition_on_img,
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
    num_step_per_epoch = num_step_per_epoch,
    grad_accum_every = args.grad_accum_every,
    learning_rate = args.lr,
    max_grad_norm = args.max_grad_norm,
    accelerator_kwargs = dict(
        cpu = False,
        step_scheduler_with_optimizer=False
    ),
    log_every_step = args.log_every_step,
    use_wandb_tracking = args.use_wandb_tracking,
    checkpoint_folder = args.model.checkpoint_folder,
    checkpoint_every_step = int(args.save_every_epoch * num_step_per_epoch),
    resume_training=args.resume_training,
    checkpoint_file_name=args.model.checkpoint_file_name,
    from_start=args.from_start,
    val_every_step=int(args.val_every_epoch * num_step_per_epoch),
)

trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg)