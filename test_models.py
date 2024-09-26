import unittest
from models import get_model
from torchview import draw_graph


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
    "senet154",
    "efficientnetb0",
    "efficientnetb1",
    "efficientnetb2",
    "efficientnetb3",
    "efficientnetb4",
    "efficientnetb5",
    "efficientnetb6",
    "efficientnetb7",
]
datasets = ["krestenitis", "sos", "chn6_cug", "cimat"]


class TestUnetEncoders(unittest.TestCase):
    pass


def create_test_for_encoder(encoder):
    def test_encoder(self):
        print(f"Testing {encoder}")
        try:
            model = get_model("unet", {"in_channels": 3, "out_channels": 1}, encoder)
            # We are using draw_graph to eval the model graph
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
        except:
            self.fail("No se pudo crear el modelo")

    return test_encoder


for encoder in encoders:
    setattr(TestUnetEncoders, f"test_{encoder}", create_test_for_encoder(encoder))


if __name__ == "__main__":
    unittest.main()
