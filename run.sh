#!/bin/bash

conda env create -f environment.yaml
conda activate yy

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29052 z_sa_3.py \
    --model_name google/gemma-2-9b-it \
    --train_set_path openai/gsm8k \
    --deepspeed ./deepspeed_configs/deepspeed_3.json\
    --max_length 256 \
    --save_every_steps 50 \
    --per_device_train_batch_size 4 \
