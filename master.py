import subprocess

architectures = ["unet"]
encoders = [
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]
datasets = ["krestenitis", "sos", "chn6_cug", "cimat"]
for model_arch in architectures:
    for model_encoder in encoders:
        for dataset in datasets:
            subprocess.run(
                [
                    "sbatch",
                    f"--job-name={model_arch}-{model_encoder}-{dataset}",
                    f"--output=outputs/{model_arch}-{model_encoder}-{dataset}-%A_%a.out",
                    "--wait",
                    "run-train.slurm",
                    dataset,
                    model_arch,
                    model_encoder,
                ]
            )
