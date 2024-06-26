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
# DATASET="mnist"

# # # Define model architecture parameters
# CHANNEL_MULT="1 2 3 4"
# MODEL_CHANNELS="16"
# ATTN_RESOLUTIONS="0"
# LAYERS_PER_BLOCK="1"


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
# python -u sample_edm.py --dataset $DATASET \
#          --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#          --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#          --sample_mode fid \
#          --num_fid_sample 5000 \
#          --fid_batch_size 1024\
#          --model_paths ~/Github/mini_edm/exps/base_mnist_20240129-1342/checkpoints \
#          --total_steps 40
       

DATASET="RAVEN10"
CHANNEL_MULT="1 2 3 4"
MODEL_CHANNELS="16"
ATTN_RESOLUTIONS="2 3"
LAYERS_PER_BLOCK="1"
## training
python -u train_edm.py --dataset $DATASET \
      --dataset_root /n/home12/binxuwang/Datasets/RAVEN_Datasets/PGM --grayscale \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 64 --num_steps 100000 \
      --learning_rate 5e-5 --accumulation_steps 2 \
      --log_step 100 --train_progress_bar \
      --save_images_step 500 \
      --save_model_iters 2000 --img_size 256


DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 100000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 25 \
      --save_images_step 500 \
      --save_model_iters 2000


DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 100000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 25 \
      --save_images_step 500 \
      --save_model_iters 2000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 100000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 25 \
      --save_images_step 500 \
      --save_model_iters 2000


# Base NN for RAVEN10_abstract
DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr Base \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching padding \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr Base \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching padding \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


# Base NN for RAVEN10_abstract
DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr BaseBlnr \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="1"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr BaseBlnr \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000
      

# Wide Deep NN for RAVEN10_abstract
DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="128"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="2"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr Wide \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 2000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="128"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="2"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr Wide \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 2000


# Wide Deep NN for RAVEN10_abstract
DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="128"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="2"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr WideBlnr \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="128"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="2"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr WideBlnr \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000
      

# Wide Deep NN for RAVEN10_abstract
DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="192"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="3"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr BigBlnr \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="192"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="3"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr BigBlnr \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="192"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="3"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr BigBlnrlrsm \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --spatial_matching bilinear \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 1e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


# Wide Deep NN for RAVEN10_abstract
DATASET="RAVEN10_abstract"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="256"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="4"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr Big \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000


DATASET="RAVEN10_abstract_onehot"
CHANNEL_MULT="1 2 4 4"
MODEL_CHANNELS="256"
ATTN_RESOLUTIONS="0 1 2"
LAYERS_PER_BLOCK="4"
## training
python -u train_abstr_edm.py --dataset $DATASET --img_size 9 \
      --exp_root $WORK_DIR/DL_Projects/mini_edm --expr Big \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 1000000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
      --save_images_step 500 \
      --save_model_iters 20000

