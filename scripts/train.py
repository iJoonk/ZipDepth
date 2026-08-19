"""
Training script for ZipDepth — monocular depth estimation.

Usage:
    # Single GPU
    python scripts/train.py --config configs/default.json
    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 scripts/train.py --config configs/default.json
    # Resume from checkpoint
    python scripts/train.py --config configs/default.json --data-root /path/to/data --resume checkpoints/epoch_3.pth

    # Warm restart (reset optimizer and scheduler)
    python scripts/train.py --config configs/default.json --data-root /path/to/data --resume checkpoints/epoch_3.pth --reset-optimizer

    # Differential LR for new layers (architecture change)
    python scripts/train.py --config configs/default.json --data-root /path/to/data --resume checkpoints/epoch_3.pth --new-layers convex_up
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import math
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from zipdepth.model.architecture import create_model
from zipdepth.data.dataset import LargeScaleDepthDataset, BalancedDomainSampler
from zipdepth.data.transforms import get_train_transforms
from zipdepth.training.trainer import ZipDepthTrainer
from zipdepth.training.distributed import (
    setup_distributed, cleanup_distributed, barrier, print_main,
    WorkerDistributedSampler, worker_init_fn, fix_state_dict_prefix,
)
from zipdepth.utils.model_utils import strip_state_dict_prefixes


# ============================================================================
# MAIN
# ============================================================================

def load_config(config_path: str) -> dict:
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        print(f"[Warning] Config not found: {config_path}")
        return {}
    with open(p) as f:
        return json.load(f)


def merge_config_args(cfg: dict, args: argparse.Namespace) -> argparse.Namespace:
    """JSON config values fill in missing CLI args. CLI always wins."""
    defaults = {
        'model_variant':       cfg.get('model', {}).get('variant', 'base'),
        'upsample_unfold':     cfg.get('model', {}).get('upsample_unfold', True),
        'alpha_ssi':           cfg.get('loss', {}).get('alpha_ssi', 1.0),
        'alpha_grad':          cfg.get('loss', {}).get('alpha_grad', 2.0),

        'height':              cfg.get('data', {}).get('height', 512),
        'width':               cfg.get('data', {}).get('width', 512),
        'index_file':          cfg.get('data', {}).get('index_file', 'data/indexes/dataset_index.json'),
        'domains':             cfg.get('data', {}).get('domains', None),
        'max_train_samples':   cfg.get('data', {}).get('max_samples', None),
        'strict_loading':      cfg.get('data', {}).get('strict_loading', None),
        'balanced_sampling':   cfg.get('data', {}).get('balanced_sampling', False),
        'balance_temperature': cfg.get('data', {}).get('balance_temperature', 0.2),
        'epochs':              cfg.get('training', {}).get('epochs', 50),
        'batch_size':          cfg.get('training', {}).get('batch_size', 16),
        'num_workers':         cfg.get('training', {}).get('num_workers', 4),
        'save_dir':            cfg.get('training', {}).get('save_dir', './checkpoints'),
        'save_every_steps':    cfg.get('training', {}).get('save_every_steps', 0),
        'max_step_checkpoints':cfg.get('training', {}).get('max_step_checkpoints', 5),
        'max_steps':           cfg.get('training', {}).get('max_steps', 0),
        'stage':               cfg.get('training', {}).get('stage', None),
        'expected_world_size': cfg.get('training', {}).get('expected_world_size', None),
        'seed':                cfg.get('training', {}).get('seed', 2027),
        'deterministic':       cfg.get('training', {}).get('deterministic', False),
        'compile_model':       cfg.get('training', {}).get('compile', True),
        'initialization':      cfg.get('training', {}).get('initialization', None),
        'init_checkpoint':     cfg.get('training', {}).get('init_checkpoint', None),
        'lr':                  cfg.get('optimizer', {}).get('lr', 5e-4),
        'weight_decay':        cfg.get('optimizer', {}).get('weight_decay', 0.01),
        'scale_lr':            cfg.get('optimizer', {}).get('scale_lr_with_gpus', False),
        'scheduler_type':      cfg.get('scheduler', {}).get('type', 'onecycle'),
        'warmup_fraction':     cfg.get('scheduler', {}).get('warmup_fraction', 0.5),
        'min_lr_ratio':        cfg.get('scheduler', {}).get('min_lr_ratio', 0.01),
        'protocol':            cfg.get('protocol', None),
        'amp':                 cfg.get('amp', {}).get('enabled', True),
        'amp_dtype':           cfg.get('amp', {}).get('dtype', 'bfloat16'),
        'wandb':               cfg.get('logging', {}).get('wandb', False),
        'profile':             cfg.get('profiler', {}).get('enabled', False),
        'profile_dir':         cfg.get('profiler', {}).get('dir', './log/profiler'),
        'profile_wait':        cfg.get('profiler', {}).get('wait', 2),
        'profile_warmup':      cfg.get('profiler', {}).get('warmup', 2),
        'profile_active':      cfg.get('profiler', {}).get('active', 5),
        'profile_repeat':      cfg.get('profiler', {}).get('repeat', 1),
    }
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_scheduler(optimizer, args, total_steps: int, steps_per_epoch: int):
    """Build the configured step-wise learning-rate schedule."""
    if total_steps <= 0 or steps_per_epoch <= 0:
        raise ValueError('Scheduler requires positive step counts')
    if args.scheduler_type == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0,
        ), None
    if args.scheduler_type != 'half_epoch_warmup_cosine':
        raise ValueError(f'Unknown scheduler type: {args.scheduler_type}')
    if not 0.0 < args.warmup_fraction <= 1.0:
        raise ValueError('warmup_fraction must be in (0, 1]')
    if not 0.0 < args.min_lr_ratio < 1.0:
        raise ValueError('min_lr_ratio must be in (0, 1)')

    warmup_steps = max(1, int(math.ceil(steps_per_epoch * args.warmup_fraction)))
    warmup_steps = min(warmup_steps, total_steps)

    def multiplier(step: int) -> float:
        # LambdaLR evaluates step=0 during construction.  Starting at one
        # warmup increment avoids a zero-LR optimizer update while reaching the
        # configured peak exactly at the end of half an epoch.
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps + 1) / remaining)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier), warmup_steps


def validate_kitti_protocol(args, world_size: int) -> None:
    """Reject accidental deviations from the frozen three-stage baseline."""
    if args.protocol != 'kitti_only_zipdepth_paper_v1':
        return
    if args.balanced_sampling:
        raise ValueError('KITTI-only protocol forbids BalancedDomainSampler')
    if args.scale_lr:
        raise ValueError('KITTI-only protocol forbids GPU-count LR scaling')
    if args.model_variant != 'base':
        raise ValueError('KITTI-only protocol requires ZipDepth-base')
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise ValueError(
            f'Expected world_size={args.expected_world_size}, got {world_size}. '
            'Use --allow-protocol-override only for smoke/overfit diagnostics.'
        )
    if args.allow_protocol_override:
        return
    expected = {
        1: (256, 10, 2e-3),
        2: (384, 5, 5e-4),
        3: (512, 3, 2.5e-4),
    }
    if args.stage not in expected:
        raise ValueError(f'KITTI-only protocol requires stage in {sorted(expected)}')
    resolution, epochs, lr = expected[args.stage]
    actual = (args.height, args.width, args.epochs, args.lr)
    wanted = (resolution, resolution, epochs, lr)
    if actual != wanted:
        raise ValueError(f'Stage {args.stage} protocol mismatch: {actual} != {wanted}')
    if args.scheduler_type != 'half_epoch_warmup_cosine':
        raise ValueError('KITTI-only protocol requires half_epoch_warmup_cosine')
    if args.warmup_fraction != 0.5 or args.min_lr_ratio != 0.01:
        raise ValueError('KITTI-only protocol requires warmup=0.5 epoch and min LR=1%')
    if args.batch_size * world_size != 96:
        raise ValueError(
            f'KITTI-only protocol requires global batch 96, got '
            f'{args.batch_size * world_size}'
        )


def main(args):
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = (rank == 0)

    cfg = load_config(args.config)
    args = merge_config_args(cfg, args)
    if args.initialization is None:
        args.initialization = (
            'resume' if args.resume else
            'checkpoint' if args.init_checkpoint else
            'random'
        )

    if args.allow_protocol_override:
        args.expected_world_size = world_size
    validate_kitti_protocol(args, world_size)

    seed = int(args.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.save_dir = args.save_dir.format(
        variant=args.model_variant,
        height=args.height,
        width=args.width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_main(f"\n{'='*80}", rank)
    print_main(f"ZipDepth Training  |  {world_size} GPU(s)  |  {device}", rank)
    print_main(f"{'='*80}", rank)

    # ========================================================================
    # SAVE EFFECTIVE CONFIG
    # ========================================================================
    if is_main:
        import time
        os.makedirs(args.save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        effective_cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        cfg_out = os.path.join(args.save_dir, f'config_{ts}.json')
        with open(cfg_out, 'w') as f:
            json.dump(effective_cfg, f, indent=2, default=str)
        print(f"[Config] Saved effective config -> {cfg_out}")

    # ========================================================================
    # MODEL
    # ========================================================================
    print_main("\nCreating model...", rank)
    model = create_model(variant=args.model_variant, upsample_unfold=args.upsample_unfold)
    if args.resume and args.init_checkpoint:
        raise ValueError('--resume and --init-checkpoint are mutually exclusive')
    if args.initialization == 'random':
        if args.resume or args.init_checkpoint:
            raise ValueError('This stage is frozen to random initialization')
        print_main('Initialization: random (no checkpoint loaded)', rank)
    elif args.initialization == 'checkpoint':
        if not args.init_checkpoint:
            raise ValueError('Checkpoint initialization requires --init-checkpoint')
        init_payload = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        init_state = init_payload.get('model_state_dict', init_payload)
        model.load_state_dict(strip_state_dict_prefixes(init_state), strict=True)
        print_main(f'Initialization: model weights from {args.init_checkpoint}', rank)
        del init_payload, init_state
    elif args.initialization != 'resume':
        raise ValueError(f'Unknown initialization mode: {args.initialization}')
    if is_main:
        model.print_model_summary()
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ========================================================================
    # DATASET
    # ========================================================================
    print_main("\nLoading training dataset...", rank)
    train_transforms = get_train_transforms(height=args.height, width=args.width)

    train_dataset = LargeScaleDepthDataset(
        index_file=args.index_file,
        domains=args.domains.split(',') if isinstance(args.domains, str) else args.domains,
        transform=train_transforms,
        max_samples=args.max_train_samples,
        strict_loading=args.strict_loading,
    )

    if args.balanced_sampling:
        train_sampler = BalancedDomainSampler(
            dataset=train_dataset,
            num_samples=len(train_dataset),
            temperature=args.balance_temperature,
            rank=rank,
            world_size=world_size,
        )
    elif is_distributed:
        train_sampler = WorkerDistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )

    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    model_params = model.module.parameters() if is_distributed else model.parameters()
    base_lr = args.lr
    actual_lr = base_lr * math.sqrt(world_size) if (is_distributed and args.scale_lr) else base_lr

    optimizer = optim.AdamW(model_params, lr=actual_lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    max_steps = getattr(args, 'max_steps', 0) or 0
    if max_steps > 0:
        total_steps = max_steps
        args.epochs = max(args.epochs, math.ceil(max_steps / steps_per_epoch))
    else:
        total_steps = steps_per_epoch * args.epochs

    scheduler, warmup_steps = build_scheduler(
        optimizer, args, total_steps=total_steps, steps_per_epoch=steps_per_epoch
    )

    # ========================================================================
    # LOGGING
    # ========================================================================
    writer = None
    if is_main and not args.wandb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'runs'))
        except Exception as e:
            print(f"[Warning] TensorBoard unavailable ({e}). Logging disabled.")

    # ========================================================================
    # TRAINER
    # ========================================================================
    trainer = ZipDepthTrainer(
        student=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=args.amp,
        log_wandb=args.wandb and is_main,
        writer=writer,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        use_profiler=args.profile,
        profile_dir=args.profile_dir,
        profile_wait=args.profile_wait,
        profile_warmup=args.profile_warmup,
        profile_active=args.profile_active,
        profile_repeat=args.profile_repeat,
        amp_dtype=args.amp_dtype,
        alpha_ssi=args.alpha_ssi,
        alpha_grad=args.alpha_grad,
        compile_model=args.compile_model,
    )

    # ========================================================================
    # PRINT CONFIG
    # ========================================================================
    print_main(f"\n{'='*80}", rank)
    print_main("TRAINING CONFIGURATION", rank)
    print_main(f"{'='*80}", rank)
    print_main(f"Model:           {args.model_variant}", rank)
    print_main(f"Loss:            SSI x{args.alpha_ssi} + Grad x{args.alpha_grad}", rank)
    print_main(f"Input size:      {args.height}x{args.width}", rank)
    print_main(f"Samples:         {len(train_dataset):,}", rank)
    print_main(f"Batches/epoch:   {len(train_loader):,}", rank)
    print_main(f"Epochs:          {args.epochs}", rank)
    print_main(f"Batch/GPU:       {args.batch_size}  (effective: {args.batch_size * world_size})", rank)
    if max_steps > 0:
        print_main(f"Max steps:       {max_steps:,}  (≈{max_steps/steps_per_epoch:.1f} epochs)", rank)
    print_main(f"Peak LR:         {actual_lr:.2e} (GPU scaling: {args.scale_lr})", rank)
    print_main(f"Scheduler:       {args.scheduler_type}", rank)
    if warmup_steps is not None:
        print_main(
            f"Warmup/decay:    {warmup_steps} steps / cosine to "
            f"{args.min_lr_ratio:.1%} of peak", rank
        )
    print_main(f"Initialization:  {args.initialization}", rank)
    print_main(f"Seed:            {args.seed} (deterministic={args.deterministic})", rank)
    print_main(f"AMP:             {args.amp} ({args.amp_dtype})", rank)
    print_main(f"{'='*80}\n", rank)

    # ========================================================================
    # RESUME
    # ========================================================================
    start_epoch = 0

    if args.resume:
        print_main(f"Resuming from: {args.resume}", rank)

        if is_distributed:
            if rank == 0:
                ckpt = torch.load(args.resume, map_location='cpu')
            else:
                ckpt = None
            barrier()
            ckpt = [ckpt]
            dist.broadcast_object_list(ckpt, src=0)
            ckpt = ckpt[0]
            barrier()
        else:
            ckpt = torch.load(args.resume, map_location=device)

        last_epoch = ckpt.get('epoch', 0)
        trainer.global_step = ckpt.get('global_step', 0)
        state_dict = fix_state_dict_prefix(ckpt.get('model_state_dict', ckpt), is_distributed)

        if args.new_layers:
            new_layer_patterns = [p.strip() for p in args.new_layers.split(',')]
            model_state = model.state_dict()
            keys_mismatch = [k for k, v in state_dict.items()
                             if k in model_state and v.shape != model_state[k].shape]
            for k in keys_mismatch:
                if is_main:
                    print(f"  Shape mismatch, skipping: {k}")
                del state_dict[k]
            missing, _ = model.load_state_dict(state_dict, strict=False)
            if is_main and missing:
                modules = sorted(set('.'.join(k.split('.')[:-1]) for k in missing))
                print(f"\nNew layers (random init): {modules}")

            raw_model = model.module if is_distributed else model
            pretrained_params, new_params = [], []
            for name, param in raw_model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(p in name for p in new_layer_patterns):
                    new_params.append(param)
                else:
                    pretrained_params.append(param)

            optimizer = optim.AdamW([
                {'params': pretrained_params, 'lr': base_lr * 0.1},
                {'params': new_params, 'lr': base_lr},
            ], weight_decay=args.weight_decay)

            start_epoch = 0
            trainer.global_step = 0
            total_steps = args.epochs * len(train_loader)
            warmup_steps = min(1000, len(train_loader) // 2)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                [optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup_steps),
                 optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_steps, base_lr * 0.01)],
                milestones=[warmup_steps],
            )
            trainer.optimizer = optimizer
            trainer.scheduler = scheduler
            print_main(f"Differential LR: pretrained @ {base_lr*0.1:.2e}, new layers @ {base_lr:.2e}", rank)

        elif args.reset_optimizer:
            model.load_state_dict(state_dict)
            start_epoch = 0
            trainer.global_step = 0
            model_params = model.module.parameters() if is_distributed else model.parameters()
            optimizer = optim.AdamW(model_params, lr=base_lr, weight_decay=args.weight_decay)
            total_steps = args.epochs * len(train_loader)
            warmup_steps = 3000
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                [optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup_steps),
                 optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_steps, base_lr * 0.01)],
                milestones=[warmup_steps],
            )
            trainer.optimizer = optimizer
            trainer.scheduler = scheduler
            print_main(f"Warm restart: LR={base_lr:.2e}, warmup={warmup_steps} steps", rank)

        else:
            # Standard resume: restore model, optimizer, and scheduler exactly.
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = trainer.global_step // len(train_loader)
            if ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print_main(f"  Scheduler state restored (last_epoch={scheduler.last_epoch})", rank)
            else:
                print_main(f"  [warn] No scheduler state in checkpoint — LR schedule reset.", rank)
            trainer.scheduler = scheduler
            print_main(f"Standard resume: epoch {start_epoch}, step {trainer.global_step}", rank)

        if is_distributed:
            barrier()

    # ========================================================================
    # TRAIN
    # ========================================================================
    try:
        trainer.train(
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            start_epoch=start_epoch,
            save_every_steps=args.save_every_steps,
            max_step_checkpoints=args.max_step_checkpoints,
            max_steps=max_steps,
        )
    except KeyboardInterrupt:
        print_main("\nInterrupted.", rank)
    finally:
        if writer:
            writer.close()
        cleanup_distributed()
        print_main("Done.", rank)


# ============================================================================
# ARGS
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ZipDepth')

    # Config
    parser.add_argument('--config', type=str, default='configs/default.json')

    # Model (override config)
    parser.add_argument('--model-variant', type=str, default=None,
                        choices=['small', 'base', 'large', 'giant'])

    # Data (required)
    parser.add_argument('--index-file', type=str, default=None)
    parser.add_argument('--domains', type=str, default=None,
                        help='Comma-separated domain filter (e.g. "kitti,nyu")')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--strict-loading', action=argparse.BooleanOptionalAction,
                        default=None)

    # Training (override config)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--save-every-steps', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Stop after this many gradient updates (overrides epoch count for scheduler too)')
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--expected-world-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--compile', dest='compile_model',
                        action=argparse.BooleanOptionalAction, default=None)

    # Resume
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--init-checkpoint', type=str, default=None,
                        help='Load model weights only (for transition to the next stage)')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Reset optimizer/scheduler on resume (warm restart)')
    parser.add_argument('--new-layers', type=str, default=None,
                        help='Comma-separated patterns for new layers (differential LR)')

    # Flags
    parser.add_argument('--amp', action='store_true', default=None)
    parser.add_argument('--amp-dtype', type=str, default=None,
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--wandb', action='store_true', default=None)
    parser.add_argument('--scale-lr', action='store_true', default=None)
    parser.add_argument('--balanced-sampling', action='store_true', default=None)
    parser.add_argument('--allow-protocol-override', action='store_true',
                        help='Allow sample/world-size/epoch overrides for diagnostics only')

    # Profiler
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--profile-dir', type=str, default=None)
    parser.add_argument('--profile-wait', type=int, default=None)
    parser.add_argument('--profile-warmup', type=int, default=None)
    parser.add_argument('--profile-active', type=int, default=None)
    parser.add_argument('--profile-repeat', type=int, default=None)

    args = parser.parse_args()
    main(args)
