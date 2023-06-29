#!/bin/bash

#SBATCH --job-name=iso_6 # Submit a job named "example"
#SBATCH --output=log_6.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --gres=gpu:1          # Use 1 GPU
#SBATCH --time=0-12:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=80000              # cpu memory size
#SBATCH --cpus-per-task=16       # cpu 개수

# source /home/${USER}/.bashrc

srun python main.py --mode train --config cifar10.train_diffusion_pl \
    --workdir work_dir6 --n_gpus_per_node 1