#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=UNet-ResNet34
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1

# srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py 17 results 5
for dataset in krestenitis chn6_cug sos cimat
do
  srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py results $dataset unet 30 --dataset_num 17 --trainset_num 01 --dataset_channels oov --model_encoder resnet34
done

