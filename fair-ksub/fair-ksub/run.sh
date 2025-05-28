#!/bin/bash
#SBATCH --job-name=vu_k-topic
#SBATCH --partition=dgx-small
#SBATCH --ntasks=16
#SBATCH --nodes=2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python ./batch_generate_all_graphs.py