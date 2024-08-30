#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=UNet-ResNet34
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1

srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py 17 results 5
