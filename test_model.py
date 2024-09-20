import argparse

from models import get_model
from torchview import draw_graph


if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture")
    parser.add_argument("--model_encoder", required=False)

    # parser.add_argument("results_path")
    # parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Generic variables
    model_arch = args.architecture
    model_encoder = args.model_encoder

    model = get_model(model_arch, {"in_channels": 3, "out_channels": 1}, model_encoder)
    if model == None:
        raise Exception("Error en la configuración del modelo")
    # Training configuration
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+efficientnetb0",
        directory="figures",
    )

    print("Done!")
