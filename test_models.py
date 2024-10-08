import pywt
import torch
import unittest
import numpy as np
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
    "cbamnet154",
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
        self.parameters = {"in_channels": 1, "out_channels": 1, "wavelets_mode": False}


class TestUnetEncoders(TestBaseEncoders):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__("unet", methodName)


class TestLinknetEncoders(TestBaseEncoders):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__("linknet", methodName)


class TestFPNEncoders(TestBaseEncoders):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__("fpn", methodName)


class TestPSPNetEncoders(TestBaseEncoders):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__("pspnet", methodName)


def create_test_for_encoder(encoder, wavelets_mode=False):
    def test_encoder(self):
        print(f"Testing {encoder}")
        try:
            self.parameters["wavelets_mode"] = wavelets_mode
            model = get_model(self.model, self.parameters, encoder)
            # We are using draw_graph to eval the model graph
            if not wavelets_mode:
                print("Test in normal mode")
                draw_graph(
                    model,
                    input_size=(1, 1, 224, 224),
                    depth=5,
                    show_shapes=True,
                    expand_nested=True,
                    save_graph=True,
                    filename=f"{self.model}-{encoder}_n",
                    directory="figures",
                )
                return
            if wavelets_mode >= 0:
                print(f"Test in wavelets mode level: {wavelets_mode}")
                x = np.random.randn(1, 1, 256, 256).astype(np.float32)
                x1, _ = pywt.dwt2(x, "db1")
                x2, _ = pywt.dwt2(x1, "db1")
                x3, _ = pywt.dwt2(x2, "db1")
                x4, _ = pywt.dwt2(x3, "db1")

                x = torch.from_numpy(x)
                x1 = torch.from_numpy(x1)
                x2 = torch.from_numpy(x2)
                x3 = torch.from_numpy(x3)
                x4 = torch.from_numpy(x4)

                # Simulate input data with wavelet decomposition
                input_data = [(x, x1, x2, x3, x4)]

                draw_graph(
                    model,
                    input_data=input_data,
                    depth=7,
                    show_shapes=True,
                    expand_nested=True,
                    save_graph=True,
                    filename=f"{self.model}-{encoder}_w{wavelets_mode}",
                    directory="figures",
                )
                return

        except Exception as e:
            self.fail(f"No se pudo crear el modelo: {e}")

    return test_encoder


for encoder in encoders:
    setattr(
        TestUnetEncoders,
        f"test_{encoder}_n",
        create_test_for_encoder(encoder, wavelets_mode=False),
    )
    setattr(
        TestLinknetEncoders,
        f"test_{encoder}_n",
        create_test_for_encoder(encoder, wavelets_mode=False),
    )
    setattr(
        TestFPNEncoders,
        f"test_{encoder}_n",
        create_test_for_encoder(encoder, wavelets_mode=False),
    )
    # setattr(TestPSPNetEncoders, f"test_{encoder}", create_test_for_encoder(encoder))

    setattr(
        TestUnetEncoders,
        f"test_{encoder}_w1",
        create_test_for_encoder(encoder, wavelets_mode=1),
    )
    setattr(
        TestLinknetEncoders,
        f"test_{encoder}_w1",
        create_test_for_encoder(encoder, wavelets_mode=1),
    )
    setattr(
        TestFPNEncoders,
        f"test_{encoder}_w1",
        create_test_for_encoder(encoder, wavelets_mode=1),
    )

if __name__ == "__main__":
    unittest.main()
