#!/bin/bash

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo " "

###########################################################################################################

# Define data set
# DATASET="cifar10"

# Define model architecture parameters
# CHANNEL_MULT="1 2 2"
# MODEL_CHANNELS="96"
# ATTN_RESOLUTIONS="16"
# LAYERS_PER_BLOCK="2"

# # Define data set
DATASET="mnist"

# # Define model architecture parameters
CHANNEL_MULT="1 2 3 4"
MODEL_CHANNELS="16"
ATTN_RESOLUTIONS="0"
LAYERS_PER_BLOCK="1"


## training
# python -u train_edm.py --dataset $DATASET \
#       --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#       --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#       --train_batch_size 512 --num_steps 500000 \
#       --learning_rate 2e-4 --accumulation_steps 1 \
#       --log_step 500 --train_progress_bar \
#        --save_images_step 5000 \
#       --save_model_iters 25000

# ## sampling
# python -u sample_edm.py --dataset $DATASET \
#         --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#         --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#         --sample_mode save \
#         --eval_batch_size 64 \
#         --model_paths \
#         --total_steps 40

# ## evaluate fid
python -u sample_edm.py --dataset $DATASET \
         --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
         --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
         --sample_mode fid \
         --num_fid_sample 5000 \
         --fid_batch_size 1024\
         --model_paths ~/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints \
         --total_steps 40
       
