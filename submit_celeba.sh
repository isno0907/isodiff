export MODEL_NAME="google/ddpm-ema-celebahq-256" #"google/ncsnpp-ffhq-256" #"google/ddpm-ema-celebahq-256" #"simlightvt/ddpm-celebahq-128"
export DATASET_PATH="/data/dataset/celeba_hq/images"
export OUTPUT_DIR="output/ddpm_celeba_hq/"

mkdir -p $OUTPUT_DIR

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=4 --main_process_port 29601 train_unconditional.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_PATH \
  --output_dir=$OUTPUT_DIR \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_epochs=5 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --split="train" \
  --lambda_iso=1e-6 \
  --normal_p=0.5 \
  --checkpointing_steps=1000 \
  --hard \
