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
                    "--job-name=TestUnet",
                    "--output=python-job-%j.out",
                    "--wait",
                    "run.slurm",
                    "krestenitis",
                ]
            )
