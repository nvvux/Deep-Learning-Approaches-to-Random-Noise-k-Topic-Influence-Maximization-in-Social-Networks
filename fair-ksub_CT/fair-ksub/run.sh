#!/bin/bash
#SBATCH --job-name=Vvux
#SBATCH --partition=small
#SBATCH --ntasks=64
#SBATCH --nodes=2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python ./test.py