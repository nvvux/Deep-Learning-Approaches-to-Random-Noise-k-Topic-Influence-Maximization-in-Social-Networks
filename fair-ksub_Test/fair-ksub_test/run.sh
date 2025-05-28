#!/bin/bash
#SBATCH --job-name=INF_k_topic
#SBATCH --partition=small
#SBATCH --ntasks=64
#SBATCH --nodes=2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python ./main.py