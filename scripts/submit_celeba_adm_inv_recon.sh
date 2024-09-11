accelerate launch --mixed_precision="fp16" --num_processes=1 --main_process_port 29504 adm_inversion_reconstruction.py  \
    --seeds=[0,1,2,3] \
    --save_path="results/inversion/" \
    --num_inference_steps=20 \
    --inversion_portion=1. \
    --unet_path="assets/weights/adm_isodiff_unet"