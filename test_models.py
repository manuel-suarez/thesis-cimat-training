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


class TestBaseEncoders(unittest.TestCase):
    def __init__(self, model, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model = model
        self.parameters = {"in_channels": 3, "out_channels": 1}


class TestUnetEncoders(TestBaseEncoders):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__("unet", methodName)


class TestLinknetEncoders(TestBaseEncoders):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__("linknet", methodName)


def create_test_for_encoder(encoder):
    def test_encoder(self):
        print(f"Testing {encoder}")
        try:
            model = get_model(self.model, self.parameters, encoder)
            # We are using draw_graph to eval the model graph
            draw_graph(
                model,
                input_size=(1, 3, 224, 224),
                depth=5,
                show_shapes=True,
                expand_nested=True,
                save_graph=True,
                filename=f"{self.model}-{encoder}",
                directory="figures",
            )
        except:
            self.fail("No se pudo crear el modelo")

    return test_encoder


for encoder in encoders:
    setattr(TestUnetEncoders, f"test_{encoder}", create_test_for_encoder(encoder))
    setattr(TestLinknetEncoders, f"test_{encoder}", create_test_for_encoder(encoder))


if __name__ == "__main__":
    unittest.main()
