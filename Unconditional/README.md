# Unconditional DomainStudio

This is the codebase the unconditional few-shot image generation with DomainStudio
Our code is based on [openai/improved-diffusion] with modifications for the unconditional few-shot image generation with the proposed DomainStudio approach.

The source models pre-trained on FFHQ and LSUN Church will be released as well.

# Environment setups

pip install -e .

Other Requirements:
Linux System \
python 3.8.13 \
pytorch 1.7.1+cu110 \
torchvision 0.8.2+cu110 \
numpy 1.23.2 \
Pillow 9.2.0 \
six 1.16.0 \
scipy 1.9.3 \
requests 2.28.1 \
mpi4py 3.0.3 \
setuptools 63.4.1 \

Other experiments are carried out on x8 NVIDIA RTX A6000 GPUs (48GB memory of each). 

# Model Parameters

All the DDPMs involved in our paper share the same model setups:

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --pair_weight 4.0 --hf_pair_weight 4.0 --hf_mse_weight 1.0"

All the datasets used in this paper have the image resolution of 256x256.

# Training
DomainStudio Training :

TRAIN_FLAGS="--lr 1e-4 --batch_size 3"

'batch_size' represents the batch size on each GPU.

## Single GPU training:
python scripts/image_train.py --data_dir /datapath $TRAIN_FLAGS $MODEL_FLAGS

## Multi-GPU training:
mpiexec -n 8 python scripts/image_train.py --data_dir /datapath $TRAIN_FLAGS $MODEL_FLAGS

# Sampling

Sampling the final output x_0:

## Single GPU sampling:
python scripts/image_sample.py --model_path /model_path/model.pt $MODEL_FLAGS

## Multi-GPU sampling:
mpiexec -n 8 python scripts/image_sample.py --model_path /model_path/model.pt $MODEL_FLAGS

Sampling several diffusion steps with interval_t (get x_T, x_T-t, x_T-2t, ..., x_0):

## Single GPU sampling:
python scripts/image_sample.py --model_path /model_path/model.pt --interval_t 200 $MODEL_FLAGS

## Multi-GPU sampling:
mpiexec -n 8 python scripts/image_sample.py --model_path /model_path/model.pt --interval_t 200 $MODEL_FLAGS
