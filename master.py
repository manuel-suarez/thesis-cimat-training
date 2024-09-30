import os
import subprocess

architectures = ["unet"]
encoders = [
    "vgg11",
]
datasets = ["cimat"]
epochs = 30
for model_arch in architectures:
    for model_encoder in encoders:
        for dataset in datasets:
            output_dir = os.path.join("outputs", model_arch, model_encoder, dataset)
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(
                [
                    "sbatch",
                    f"--job-name={model_arch}-{model_encoder}-{dataset}",
                    f"--output=outputs/{model_arch}/{model_encoder}/{dataset}/train-%A_%a.out",
                    "--wait",
                    "run-train.slurm",
                    dataset,
                    model_arch,
                    model_encoder,
                    str(epochs),
                ]
            )
