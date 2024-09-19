#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=UNet-ResNet152
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1

# srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py 17 results 5
dataset=$1
model_arch=$2
model_encoder=$3
echo Parámetros:
echo $model_arch
echo $model_encoder
echo $dataset
# for dataset in krestenitis chn6_cug sos cimat
# do 
if [[ $dataset == "cimat" ]]
then
  srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py results $dataset $model_arch 1 --dataset_num 17 --trainset_num 01 --dataset_channels oov --model_encoder $model_encoder
else
  srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py results $dataset $model_arch 1 --model_encoder $model_encoder
fi
# done

