# Text-to-Image DomainStudio

This is the codebase for text-to-image few-shot image generation with DomainStudio
Our code is implemented based on [huggingface/diffusers] to realize training and generation with Stable Diffusion.
Our code is also based on the implementation of DreamBooth from diffusers.

# Environment setups
The same as the environment of huggingface/diffusers.

# DomainStudio Training
## Setting Variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export INSTANCE_DIR="path-to-instance-images"

export CLASS_DIR="path-to-class-images"

export OUTPUT_DIR="path-to-save-model"

## Training
accelerate launch train_domainstudio.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --image_loss_weight=1e+2 \
  --hf_loss_weight=1e+2 --hfmse_loss_weight=0.1 \
  --instance_prompt="instance prompt" \  # eg., a [V] dog
  
  --class_prompt="class prompt" \ # eg., a dog
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub

# Personalized DomainStudio Training
## Setting Variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export INSTANCE_DIR="path-to-instance-images"

export STYLE_DIR="path-to-style_images"

export CLASS_DIR="path-to-class-images"

export OUTPUT_DIR="path-to-save-model"

## Training
accelerate launch train_domainstudio.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --style_data_dir=$STYLE_DIR \ 
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --image_loss_weight=1e+2 \
  --hf_loss_weight=1e+2 --hfmse_loss_weight=0.1 \
  --instance_prompt="instance prompt" \  # eg., a [V] dog
  
  --style_prompt="style prompt" \ # eg., a [V] dog in the [S] style
  
  --class_prompt="class prompt" \ # eg., a dog
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub


