import os
import sys
from argparse import ArgumentParser

# Import custom modules: model, dataset, trainer, and config utils
from src.vae.vae_wireframe import AutoencoderKLWireframe as MyModel
from src.dataset.dataset import WireframeDataset as MyDataset
from src.trainer.trainer_vae import Trainer as MyTrainer
from src.utils.config import NestedDictToClass, load_config

# Arguments
# Parse command-line arguments
parser = ArgumentParser(description='Train wireframe vae model.')
parser.add_argument('--config', type=str, default='', help='Path to config file.')
parser.add_argument('--curve_vae_config', required=False, type=str, default='', help='Path to config file.')
program_args = parser.parse_args()

# Load and wrap config as attribute-accessible object
cfg = load_config(program_args.config)
args = NestedDictToClass(cfg)

isDebug = True if sys.gettrace() else False

if isDebug:
    args.use_wandb_tracking = False
    args.batch_size = 1
    args.num_workers = 1


train_dataset = MyDataset(
    dataset_file_path = args.data.train_set_file_path,
    replication=args.data.replication,
    max_num_lines=args.model.max_curves_num,
    is_train=True)

val_dataset = MyDataset(
    dataset_file_path = args.data.val_set_file_path,
    max_num_lines=args.model.max_curves_num,
    is_train=False,
)   

model = MyModel(
    latent_channels=args.model.latent_channels,
    attn_encoder_depth=args.model.attn_encoder_depth,
    attn_decoder_self_depth=args.model.attn_decoder_self_depth,
    attn_decoder_cross_depth=args.model.attn_decoder_cross_depth,
    attn_dim=args.model.attn_dim,
    num_heads=args.model.num_heads,
    max_row_diff=args.model.max_row_diff,
    max_col_diff=args.model.max_col_diff,
    max_curves_num=args.model.max_curves_num,
    wireframe_latent_num=args.model.wireframe_latent_num,
    label_smoothing=args.label_smoothing,
    cls_loss_weight=args.model.cls_loss_weight,
    segment_loss_weight=args.model.segment_loss_weight,
    col_diff_loss_weight=args.model.col_diff_loss_weight,
    row_diff_loss_weight=args.model.row_diff_loss_weight,
    curve_latent_loss_weight=args.model.curve_latent_loss_weight,
    kl_loss_weight=args.model.kl_loss_weight,
    curve_latent_embed_dim=args.model.curve_latent_embed_dim,
    use_mlp_predict=args.model.use_mlp_predict,
    use_latent_pos_emb=args.model.use_latent_pos_emb,
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
    checkpoint_every_step = args.save_every_epoch * num_step_per_epoch,
    resume_training=args.resume_training,
    checkpoint_file_name=args.model.checkpoint_file_name,
    from_start=args.from_start,
    val_every_step=int(args.val_every_epoch * num_step_per_epoch),
)

# Launch training
trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg)
