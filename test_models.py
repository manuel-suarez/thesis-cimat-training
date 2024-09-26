from logging import exception
import unittest
from models import get_model


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
            _ = get_model("unet", {"in_channels": 3, "out_channels": 1}, encoder)
        except:
            self.fail("No se pudo crear el modelo")

    return test_encoder


for encoder in encoders:
    setattr(TestUnetEncoders, f"test_{encoder}", create_test_for_encoder(encoder))


if __name__ == "__main__":
    unittest.main()
