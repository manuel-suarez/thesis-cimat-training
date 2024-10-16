import os
import subprocess

architectures = ["unet"]
encoders = ["vgg11", "resnet18"]
datasets = ["cimat"]
wavelets_modes = ["no", "1", "2", "3"]
epochs = 1
for model_arch in architectures:
    for model_encoder in encoders:
        for wavelets_mode in wavelets_modes:
            for dataset in datasets:
                output_dir = os.path.join(
                    "outputs",
                    model_arch,
                    model_encoder,
                    f"wavelets_{wavelets_mode}",
                    dataset,
                )
                os.makedirs(output_dir, exist_ok=True)
                subprocess.run(
                    [
                        "sbatch",
                        f"--job-name={model_arch}-{model_encoder}-{dataset}",
                        f"--output=outputs/{model_arch}/{model_encoder}/wavelets_{wavelets_mode}/{dataset}/train-%A_%a.out",
                        "--wait",
                        "run-train.slurm",
                        dataset,
                        model_arch,
                        model_encoder,
                        str(epochs),
                        wavelets_mode,
                    ]
                )
