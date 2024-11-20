#!/bin/bash

# We need to loop over differents architectures, encoders and wavelets mode
# To avoid exhaut disk space limit we will be transfer end training results to siimon5 server
set -x
set -e
dataset=sos 
epochs=30
remote_dest=distributed_training
for model_arch in unet; do
  for model_encoder in vgg11 vgg13 resnet18 resnet34; do
    for wavelets_mode in no; do
      for dataset in sos krestenitis; do
        output_dir=outputs/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset
        results_dir=results/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset
        dest_dir=dest/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset
        mkdir -p $output_dir
        mkdir -p $results_dir
        mkdir -p $dest_dir
        # Run sbatch training process
        sbatch --job-name=${model_arch}-${model_encoder}-${dataset} --output=outputs/$model_arch/$model_encoder/wavelets_${wavelets_mode}/${dataset}/train-%A_%a.out run-train.slurm $dataset $model_arch $model_encoder $epochs $wavelets_mode
        # Do an active wait
        while [ ! -f dest/$model_arch/$model_encoder/wavelets_${wavelets_mode}/${dataset}/output.txt ]; do
          # Sleep
          echo "dest/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset result not created, waiting 30m..."
          sleep 5m
        done
        echo "dest/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset created, proceeding to move to siimon5"
        # Move outputs and results
        scp -r -P 2235 outputs/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset manuelsuarez@siimon5.cimat.mx:/home/mariocanul/image_storage/ssh_sharedir/$remote_dest/outputs_${model_arch}_${model_encoder}_wavelets_${wavelets_mode}_${dataset}
        scp -r -P 2235 results/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset manuelsuarez@siimon5.cimat.mx:/home/mariocanul/image_storage/ssh_sharedir/$remote_dest/results_${model_arch}_${model_encoder}_wavelets_${wavelets_mode}_${dataset}
        # Delete local files to avoid exceed disk quota
        rm -rf dest/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset
        rm -rf outputs/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset
        rm -rf results/$model_arch/$model_encoder/wavelets_${wavelets_mode}/$dataset
      done
    done
  done
done
