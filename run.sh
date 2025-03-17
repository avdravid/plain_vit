#!/bin/bash

# Path to the main script
MUPVIT_MAIN=main.py

# Resource configuration
N_WORKERS=80 #32
N_THREADS=124 #64

# Set environment variables
export NUMEXPR_MAX_THREADS=$N_THREADS

# Make all 8 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL settings to help with timeout issues
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1200
export NCCL_SOCKET_IFNAME=lo
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# Run with multiprocessing distributed
python $MUPVIT_MAIN /home/amil/ilsvrc_2024-01-04_1913 \
    --workers $N_WORKERS \
    --epochs 450 \
    --batch-size 1024 \
    --report-to wandb \
    --name prune_50model_600epochs\
    --seed 0 \
    --multiprocessing-distributed \
    --dist-url "tcp://localhost:8888" \
    --world-size 1 \
    --rank 0\
  