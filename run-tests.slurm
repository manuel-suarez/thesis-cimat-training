#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=TestModels
#SBATCH --time=0
#SBATCH --mem=0

srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python -m unittest -v test_models
