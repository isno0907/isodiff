#!/bin/bash

#SBATCH --job-name=iso_3 # Submit a job named "example"
#SBATCH --output=log_3.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=mlvu
#SBATCH --gres=gpu:4          # Use 1 GPU
#SBATCH --time=2-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=100000              # cpu memory size
#SBATCH --cpus-per-task=16       # cpu 개수

# source /home/${USER}/.bashrc

srun python main.py --mode train --config cats.train_diffusion_pl \
    --workdir work_dir3 --n_gpus_per_node 4