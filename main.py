# Modified from https://github.com/pytorch/examples/blob/main/imagenet/main.py
import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"


import argparse
import functools
import math
import random
import shutil
import numpy as np
import time
import warnings
from datetime import datetime
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

# Import torch.cuda.amp for mixed precision training
#from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Subset


import schedulefree
import wandb

from simple_vit import SimpleVisionTransformer
from transforms import TwoHotMixUp, TFInceptionCrop, RandAugment17


# "(...)/python3.10/site-packages/torch/_inductor/compile_fx.py:140: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
torch.set_float32_matmul_precision('high')

def estimate_time_to_completion(current_step, total_steps, batch_time_avg):
    """
    Estimate the time remaining for training.
    
    Args:
        current_step (int): Current training step
        total_steps (int): Total number of training steps
        batch_time_avg (float): Average time per batch
    
    Returns:
        str: Estimated time to completion in a human-readable format
    """
    remaining_steps = total_steps - current_step
    estimated_seconds = remaining_steps * batch_time_avg
    
    # Convert seconds to hours, minutes, seconds
    hours, remainder = divmod(int(estimated_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--prefetch-factor', default=1, type=int, metavar='N',
                    help='number of batches for each worker to prefetch (default: 1)')
parser.add_argument('--hidden-dim', default=384, type=int, metavar='N',
                    help='Embedding dimension of the ViT (default: 384)')
parser.add_argument('--input-resolution', default=224, type=int, metavar='RES',
                    help='Input resolution, i.e. train/val crop size (default: 224)')
parser.add_argument('--patch-size', default=16, type=int, metavar='PS')
parser.add_argument('--num-layers', default=12, type=int, metavar='N')
parser.add_argument('--num-heads', default=6, type=int, metavar='N')
parser.add_argument('--posemb', default='sincos2d', type=str,
                    choices=['none', 'sincos2d', 'learn'])
parser.add_argument('--mlp-head', action='store_true',
                    help='Use a MLP classification head with one hidden tanh layer '
                         'instead of a single linear layer')
parser.add_argument('--representation-size', default=None, type=int, metavar='N',
                    help='Size of the MLP classification head hidden layer, '
                         "defaults to --hidden-dim. No effect if --mlp-head isn't set")
parser.add_argument('--pool-type', default='gap', type=str, choices=['gap', 'tok'])
parser.add_argument('--register', default=0, type=int, metavar='N',
                    help='Number of registers (additional tokens), see '
                         'https://arxiv.org/abs/2309.16588')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-steps', default=2500, type=int, metavar='N',
                    help='eval and log every N steps')
parser.add_argument('--log-epoch', nargs='*', default=[], type=int,
                    help='eval and log at the specified epochs.')
parser.add_argument('--start-step', default=0, type=int, metavar='N',
                    help='manual step number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--accum-freq", default=1, type=int,
                    help="Update the model every --acum-freq steps.")
parser.add_argument('--schedule-free', action='store_true',
                    help='Use schedule-free AdamW optimizer (https://arxiv.org/abs/2405.15682).')
parser.add_argument("--warmup", default=10000, type=int,
                    help="Number of steps to warmup for.")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='maximum learning rate', dest='lr')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for AdamW')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for AdamW')
parser.add_argument('--polynomial-weighting-power', default=0.0, type=float, metavar='r',
                    help='Use polynomial weighting in the average with power r '
                         'for schedule-free AdamW (default 0.0)')
parser.add_argument('--decoupled-weight-decay', action='store_true', default=True,
                    help='Run weight decay as it is w/o multiplying by LR. '
                         'See https://fabian-sp.github.io/posts/2024/02/decoupling/')
parser.add_argument('--no-decoupled-weight-decay', action='store_false', dest='decoupled_weight_decay',
                    help='Do not run weight decay as it is w/o multiplying by LR')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                    help="Max norm for gradient clip (default: 1.0)")
parser.add_argument('--torchvision-inception-crop', action='store_true',
                    help="Switch back to torchvision's RandomResizedCrop(), "
                         'which actually improves the model')
parser.add_argument('--lower-scale', type=float, default=0.05,
                    help="Lower bound of the area of the Inception crop (default: 0.05)")
parser.add_argument('--upper-scale', type=float, default=1.0,
                    help="Upper bound of the area of the Inception crop (default: 1.0)")
parser.add_argument('--mixup-alpha', default=0.2, type=float,
                    help='Beta distribution shape parameter for the MixUp (default: 0.2). '
                         'Use 0.0 to turn MixUp off.')
parser.add_argument('--randaug', action='store_true', default=True,
                    help='Use RandAug (default: True)')
parser.add_argument('--no-randaug', action='store_false', dest='randaug',
                    help='Do not use RandAug')
parser.add_argument("--randaug-magnitude", default=10, type=int)
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fake-data', action='store_true', help="use fake data to benchmark")
parser.add_argument("--logs", default="./logs/", type=str,
                    help="Where to store logs. Use None to avoid storing logs.")
parser.add_argument('--name', default=None, type=str,
                    help='Optional identifier for the experiment when storing logs. '
                         'Otherwise use current time.')
parser.add_argument("--report-to", default='', type=str,
                    help="Options are ['wandb']")
parser.add_argument("--wandb-notes", default='', type=str,
                    help="Notes if logging with wandb")
# Add AMP related arguments
parser.add_argument('--amp', action='store_true', default=True,
                    help='Use automatic mixed precision training')
parser.add_argument('--no-amp', action='store_false', dest='amp',
                    help='Do not use automatic mixed precision training')
parser.add_argument('--amp-dtype', type=str, default='bfloat16',
                    choices=['float16', 'bfloat16'],
                    help='Data type to use with AMP (default: float16)')
best_acc1 = 0


def collate(batch, mixup):
    return mixup(*torch.utils.data.default_collate(batch))


def chunk(n, device, *tensors):
    tensors = tuple(t.to(device=device, non_blocking=True) for t in tensors)
    if n == 1:
        yield tensors
    else:
        yield from zip(*(t.chunk(n) for t in tensors))


def main():
    args = parser.parse_args()
    
    if not args.mlp_head:
        args.representation_size = None
    elif args.representation_size is None:
        args.representation_size = args.hidden_dim

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # cudnn.deterministic = True
        # cudnn.benchmark = False
        cudnn.benchmark = True
        


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        args.ngpus_per_node = torch.cuda.device_count()
        if args.ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        args.ngpus_per_node = 1

    # get the name of the experiments
    if args.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = '-'.join([
            date_str,
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"amp_{args.amp}"  # Add AMP info to experiment name
        ])

    log_base_path = os.path.join(args.logs, args.name)
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.wandb = 'wandb' in args.report_to

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args)


def is_primary(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + gpu
    if args.distributed or args.ngpus_per_node > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = SimpleVisionTransformer(
        image_size=args.input_resolution,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.hidden_dim * 2,       #args.hidden_dim*2 for pruning 50%
        posemb=args.posemb,
        representation_size=args.representation_size,
        pool_type=args.pool_type,
        register=args.register,
    )

    wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
    non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]
    args.total_batch_size = args.batch_size

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
        device = torch.device("cpu")
        # AMP not supported on CPU, disable it
        args.amp = False
    elif torch.cuda.is_available():
        model.cuda()
        device = torch.device("cuda")
        if args.distributed or args.ngpus_per_node > 1:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available
            if args.gpu is not None:
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / args.ngpus_per_node)
                args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        device = torch.device("mps")
        model = model.to(device)
        # AMP has limited support on MPS, disable it if not using CUDA
        args.amp = False

    if args.decoupled_weight_decay:
        args.weight_decay /= args.lr

    params = [
        {"params": wd_params, "weight_decay": args.weight_decay},
        {"params": non_wd_params, "weight_decay": 0.},
    ]

    if args.schedule_free:
        optimizer = schedulefree.AdamWScheduleFree(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            warmup_steps=args.warmup,
            r=args.polynomial_weighting_power,
        )
        optimizer.train()
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2)
        )

    # Initialize the GradScaler for AMP
    scaler = GradScaler() if args.amp else None
    
    # Set AMP dtype based on user choice
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
    
    # Data loading code
    if args.fake_data:
        print("=> Fake data is used!")
        input_shape = (3, args.input_resolution, args.input_resolution)
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        train_dataset = datasets.FakeData(1281167, input_shape, 1000, transform)
        val_dataset = datasets.FakeData(50000, input_shape, 1000, transform)
    else:
        value_range = v2.Normalize(
            mean=[0.5] * 3,
            std=[0.5] * 3)

        cutout_const = 40
        translate_const = 100
        MAX_LEVEL = 10

        translate_magnitude = lambda num_bins, _h, _w: torch.linspace(0.0, translate_const, num_bins)
        shear_magnitude = lambda num_bins, _h, _w: torch.linspace(0.0, 0.3, num_bins)
        enhance_magnitude = lambda num_bins, _h, _w: torch.linspace(0, 0.9, num_bins)  # It was -0.9, 0.9 but negative magnitude results in the opposite effect.

        RandAugment17._AUGMENTATION_SPACE = {
            "TranslateX": (translate_magnitude, True),
            "TranslateY": (translate_magnitude, True),
            "ShearX": (shear_magnitude, True),
            "ShearY": (shear_magnitude, True),
            "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (enhance_magnitude, False),
            "Color": (enhance_magnitude, False),
            "Contrast": (enhance_magnitude, False),
            "Sharpness": (enhance_magnitude, False),
            "Posterize": (  # Unchanged
                lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
                False,
            ),
            "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),  # Unchanged
            "AutoContrast": (lambda num_bins, height, width: None, False),  # Unchanged
            "Equalize": (lambda num_bins, height, width: None, False),  # Unchanged
            "Invert": (lambda num_bins, height, width: None, False),  # "New" (equivalent to MAX_LEVEL Solarize)
            "SolarizeAdd": (lambda num_bins, height, width: torch.linspace(0., 110., num_bins), False),  # New
            "Cutout": (lambda num_bins, height, width: torch.linspace(0., float(cutout_const), num_bins), False),  # New
        }
        randaug = RandAugment17(2, args.randaug_magnitude, num_magnitude_bins=MAX_LEVEL + 1, fill=[128] * 3)
        inception_crop = v2.RandomResizedCrop if args.torchvision_inception_crop else TFInceptionCrop

        transform = [
            v2.ToImage(),
            inception_crop(args.input_resolution, scale=(args.lower_scale, args.upper_scale)),
            v2.RandomHorizontalFlip()
        ]
        if args.randaug:
            transform.append(randaug)
        transform.extend([
            v2.ToDtype(torch.float32, scale=True),
            value_range
        ])

        train_dataset = datasets.ImageNet(args.data, split='train', transform=v2.Compose(transform))

        val_dataset = datasets.ImageNet(
            args.data,
            split='val',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(args.input_resolution),
                v2.ToDtype(torch.float32, scale=True),
                value_range,
            ]))

    n = len(train_dataset)
    total_steps = round(n * args.epochs / args.total_batch_size)
    args.specified_steps = {round(n * epoch / args.total_batch_size) for epoch in args.log_epoch}
    args.specified_steps.add(total_steps)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    mixup = TwoHotMixUp(alpha=args.mixup_alpha, prefetch_factor=args.prefetch_factor, batch_size=args.batch_size)
    collate_fn = functools.partial(collate, mixup=mixup)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.prefetch_factor * args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        collate_fn=collate_fn, drop_last=True, multiprocessing_context='spawn',
        prefetch_factor=1, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.prefetch_factor * args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        multiprocessing_context='spawn', prefetch_factor=1)

    if args.schedule_free:
        scheduler = None
    else:
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step / args.warmup)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [args.warmup])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            args.start_step = checkpoint['step']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.schedule_free:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint and args.amp:
                scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.resume, checkpoint['step']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.wandb and is_primary(args):
        wandb.init(
            project="mup-vit",
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto',
            config=vars(args),
        )
        params_file = os.path.join(args.logs, args.name, "params.txt")
        wandb.save(params_file)

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    print('Compiling model...')
    original_model = model

    model = torch.compile(original_model)

    # model = torch.compile(
    #     original_model, backend= "inductor", mode="reduce-overhead")  


    if args.evaluate:
        # evaluate on validation set.
        validate(val_loader, model, args.start_step, device, args)
        return

    train(train_loader, train_sampler, val_loader, args.start_step, total_steps, 
          original_model, model, optimizer, scheduler, device, args, scaler, amp_dtype)


def infinite_loader(loader, sampler):
    epoch = 0
    while True:
        if sampler:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def train(train_loader, train_sampler, val_loader, start_step, total_steps, 
          original_model, model, optimizer, scheduler, device, args, scaler, amp_dtype):
    batch_time = AverageMeter('Time', device, ':6.3f')
    data_time = AverageMeter('Data', device, ':6.3f')
    losses = AverageMeter('Loss', device, ':.4e')
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, losses])

    # switch to train mode
    model.train()
    end = time.time()
    best_acc1 = 0
    # generator moves data to the same device as model
    gen = (
        (images, lam, target1, target2) for (img, lam, trt1, trt2) in infinite_loader(train_loader, train_sampler) for images, target1, target2 in zip(
            img.to(device, non_blocking=True),
            trt1.to(device, non_blocking=True),
            trt2.to(device, non_blocking=True))
    )

    for step, (images, lam, target1, target2) in zip(range(start_step + 1, total_steps + 1), gen):
        # measure data loading time
        data_time.update(time.time() - end)
        step_loss = 0.0

        for img, trt1, trt2 in chunk(args.accum_freq, device, images, target1, target2):
            # Use autocast for mixed precision where supported
            if args.amp:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    # compute output
                    _, loss = model(img, lam, trt1, trt2)
                    # Scale loss for gradient accumulation
                    loss = loss / args.accum_freq
                
                # record loss
                step_loss += loss.item() * args.accum_freq
                
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
            else:
                # compute output with full precision
                _, loss = model(img, lam, trt1, trt2)
                
                # record loss
                step_loss += loss.item()
                
                # compute gradient
                (loss / args.accum_freq).backward()

        losses.update(step_loss, images.size(0))

        # do SGD step
        if args.amp:
            # Unscales gradients and clips as needed
            scaler.unscale_(optimizer)
            l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
            # Update parameters and update the scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            progress.display(step)
            
            # Estimate time to completion
            time_to_completion = estimate_time_to_completion(step, total_steps, batch_time.avg)
            print(f"Estimated time to completion: {time_to_completion}")

            if args.wandb and is_primary(args):
                with torch.no_grad():
                    l2_params = sum(p.square().sum().item() for _, p in model.named_parameters())

                samples_per_second_per_gpu = args.batch_size / batch_time.val
                samples_per_second = samples_per_second_per_gpu * args.world_size
                log_data = {
                    "train/loss": step_loss,
                    "data_time": data_time.val,
                    "batch_time": batch_time.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "l2_grads": l2_grads.item(),
                    "l2_params": math.sqrt(l2_params),
                    "time_to_completion": time_to_completion
                }
                if scheduler:
                    log_data["lr"] = scheduler.get_last_lr()[0]
                # Add AMP scale factor to logs if using AMP
                if args.amp:
                    log_data["amp_scale"] = scaler.get_scale()
                wandb.log(log_data, step=step)
        if step % args.log_steps == 0 or step in args.specified_steps:
            if args.schedule_free:
                optimizer.eval()
                acc1 = validate(val_loader, model, step, device, args)
                optimizer.train()
            else:
                acc1 = validate(val_loader, model, step, device, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_primary(args):
                ckpt = {
                    'step': step,
                    'state_dict': original_model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }
                if scheduler:
                    ckpt['scheduler'] = scheduler.state_dict()
                if args.amp and scaler is not None:
                    ckpt['scaler'] = scaler.state_dict()
                # Always save a step-specific checkpoint whenever validation occurs
                save_checkpoint(ckpt, is_best, args.checkpoint_path, step=step)

        if scheduler:
            scheduler.step()

def validate(val_loader, model, step, device, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            torch.cuda.empty_cache()
            end = time.time()
            # generator moves data to the same device as model
            gen = (b for images, target in loader for b in chunk(args.prefetch_factor, device, images, target))
            for i, (images, target) in enumerate(gen):
                i = base_progress + i
                for img, trt in chunk(args.accum_freq, device, images, target):
                    # # compute output
                    if args.amp:
                        with autocast(device_type=device.type, 
                                     dtype=torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16):
                            output, loss = model(img, 1.0, trt, trt)
                    else:
                        output, loss = model(img, 1.0, trt, trt)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, trt, topk=(1, 5))
                    losses.update(loss.item(), img.size(0))
                    top1.update(acc1[0].item(), img.size(0))
                    top5.update(acc5[0].item(), img.size(0))
                    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

    batch_time = AverageMeter('Time', device, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', device, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', device, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', device, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        args.prefetch_factor * len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, multiprocessing_context='spawn')
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    if args.wandb and is_primary(args):
        log_data = {
            'val/loss': losses.avg,
            'val/acc@1': top1.avg,
            'val/acc@5': top5.avg,
        }
        wandb.log(log_data, step=step)

    return top1.avg


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar', step=None):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if step is not None:
        shutil.copyfile(filename, os.path.join(path, f'model_step_{step}.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, device, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':
    main()