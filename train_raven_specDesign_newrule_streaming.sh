#!/bin/bash
#SBATCH -t 54:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=64G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 14,15
#SBATCH -o UNet_RAVEN_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e UNet_RAVEN_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--expr WideBlnrX3_new_stream16M       --dataset RAVEN10_abstract        --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
 --expr BigBlnrX3_new_stream16M        --dataset RAVEN10_abstract        --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
'
# --expr BaseX3_new           --dataset RAVEN10_abstract        --layers_per_block 1 --model_channels 64  --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching padding --learning_rate 2e-4
# --expr BaseX3_new           --dataset RAVEN10_abstract_onehot --layers_per_block 1 --model_channels 64  --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching padding --learning_rate 2e-4
# --expr WideX3_new           --dataset RAVEN10_abstract        --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching padding --learning_rate 2e-4
# --expr WideX3_new           --dataset RAVEN10_abstract_onehot --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching padding --learning_rate 2e-4
# --expr BaseBlnrX3_new       --dataset RAVEN10_abstract        --layers_per_block 1 --model_channels 64  --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BaseBlnrX3_new       --dataset RAVEN10_abstract_onehot --layers_per_block 1 --model_channels 64  --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr WideBlnrX3_new       --dataset RAVEN10_abstract        --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr WideBlnrX3_new       --dataset RAVEN10_abstract_onehot --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BigBlnrX3_new        --dataset RAVEN10_abstract        --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BigBlnrX3_new        --dataset RAVEN10_abstract_onehot --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BBigBlnrX3_new       --dataset RAVEN10_abstract        --layers_per_block 4 --model_channels 256 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BBigBlnrX3_new       --dataset RAVEN10_abstract_onehot --layers_per_block 4 --model_channels 256 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4 
# --expr BBigBlnrX3_new       --dataset RAVEN10_abstract_onehot --layers_per_block 4 --model_channels 256 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 1e-4 
# --expr WideBlnrX3_new       --dataset RAVEN10_abstract        --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr WideBlnrX3_new       --dataset RAVEN10_abstract        --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 0   --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BigBlnrX3_new        --dataset RAVEN10_abstract        --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr BigBlnrX3_new        --dataset RAVEN10_abstract_onehot --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4
# --expr WideBlnrX3_new_noattn --dataset RAVEN10_abstract        --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 0   --train_batch_size 256 --spatial_matching bilinear --learning_rate 2e-4


export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
conda deactivate
mamba activate torch2
which python
which python3
# run code
cd /n/home12/binxuwang/Github/mini_edm

## training
python -u train_abstr_edm_RAVEN_stream.py --exp_root $STORE_DIR/DL_Projects/mini_edm --img_size 9 \
    --train_attr_fn attr_all_1000k.pt --cmb_per_class 400000 \
    --dataset_root $STORE_DIR/Datasets/RPM_dataset/RPM1000k \
    --log_step 100 --train_progress_bar  --eval_batch_size 2048 \
    --save_images_step 2500 --save_model_iters 20000 --num_steps 1000000 \
    --accumulation_steps 1 \
    $param_name        

