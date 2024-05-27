export MODEL_NAME="CompVis/ldm-celebahq-256" #"google/ncsnpp-ffhq-256" #"google/ddpm-ema-celebahq-256" #"simlightvt/ddpm-celebahq-128"
export DATASET_NAME="/data/dataset/celeba_hq/images" #"Norod78/Vintage-Faces-FFHQAligned" "cr7Por/ffhq_controlnet_5_2_23"
export OUTPUT_DIR="scratch/sd_scratch"

mkdir -p $OUTPUT_DIR

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=4 --main_process_port 29502 train_unconditional_ldm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_PATH \
  --output_dir=$OUTPUT_DIR \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=8\
  --num_epochs=5 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --split="train" \
  --lambda_iso=1e-03 \
  --normal_p=0.5 \
  --checkpointing_steps=1000 \
  --subfolder \
