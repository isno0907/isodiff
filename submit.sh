#!/bin/bash

#SBATCH --job-name=iso_1 # Submit a job named "example"
#SBATCH --output=log_1.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=mlvu
#SBATCH --gres=gpu:4          # Use 1 GPU
#SBATCH --time=2-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=100000              # cpu memory size
#SBATCH --cpus-per-task=16       # cpu 개수

# source /home/${USER}/.bashrc

# srun python main.py --mode train --config cats.train_diffusion \
#     --workdir work_dir1 --n_gpus_per_node 4

srun python compute_fid_statistics.py --path data/processed/cats.zip --file cats.npz