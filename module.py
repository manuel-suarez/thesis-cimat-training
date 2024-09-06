import lightning as L
import torch
import torchmetrics
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor


class CimatModule(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn, metrics_mode="binary", num_classes=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # Training metrics
        self.train_sensitivity = torchmetrics.Recall(
            task=metrics_mode, num_classes=num_classes
        )
        self.train_specificity = torchmetrics.Specificity(
            task=metrics_mode, num_classes=num_classes
        )
        self.train_accuracy = torchmetrics.Accuracy(
            task=metrics_mode, num_classes=num_classes
        )
        self.train_f1score = torchmetrics.F1Score(
            task=metrics_mode, num_classes=num_classes
        )
        self.train_meaniou = torchmetrics.JaccardIndex(
            task=metrics_mode, num_classes=num_classes
        )
        # Validation metrics
        self.valid_sensitivity = torchmetrics.Recall(
            task=metrics_mode, num_classes=num_classes
        )
        self.valid_specificity = torchmetrics.Specificity(
            task=metrics_mode, num_classes=num_classes
        )
        self.valid_accuracy = torchmetrics.Accuracy(
            task=metrics_mode, num_classes=num_classes
        )
        self.valid_f1score = torchmetrics.F1Score(
            task=metrics_mode, num_classes=num_classes
        )
        self.valid_meaniou = torchmetrics.JaccardIndex(
            task=metrics_mode, num_classes=num_classes
        )
        # Test metrics
        self.test_sensitivity = torchmetrics.Recall(
            task=metrics_mode, num_classes=num_classes
        )
        self.test_specificity = torchmetrics.Specificity(
            task=metrics_mode, num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task=metrics_mode, num_classes=num_classes
        )
        self.test_f1score = torchmetrics.F1Score(
            task=metrics_mode, num_classes=num_classes
        )
        self.test_meaniou = torchmetrics.JaccardIndex(
            task=metrics_mode, num_classes=num_classes
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # print("\n -= TRAINING =-\n")
        # print("Inputs: ", inputs.shape, labels.shape)
        # print("Types: ", inputs.type(), labels.type())
        outputs = self.model(inputs)
        # print("Outputs: ", outputs.shape)
        # print("Type: ", outputs.type())
        # preds = torch.argmax(outputs, dim=1)
        # print("Predictions: ", preds.shape)
        # print("Type: ", preds.type())
        train_loss = self.loss_fn(outputs, labels)
        self.train_sensitivity(outputs, labels)
        self.train_specificity(outputs, labels)
        self.train_accuracy(outputs, labels)
        self.train_f1score(outputs, labels)
        self.train_meaniou(outputs, labels)
        self.log_dict(
            {
                "train_loss": train_loss,
                "train_sensitivity": self.train_sensitivity,
                "train_specificity": self.train_specificity,
                "train_accuracy": self.train_accuracy,
                "train_f1score": self.train_f1score,
                "train_meaniou": self.train_meaniou,
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # print("\n -= VALIDATION =-\n")
        # print("Inputs: ", inputs.shape, labels.shape)
        # print("Types: ", inputs.type(), labels.type())
        outputs = self.model(inputs)
        # print("Outputs: ", outputs.shape)
        # print("Type: ", outputs.type())
        # preds = torch.argmax(outputs, dim=1)
        # #print("Predictions: ", preds.shape)
        # #print("Type: ", preds.type())
        # valid_loss = self.loss_fn(outputs, labels)
        valid_loss = F.cross_entropy(outputs, labels, reduction="mean")
        self.valid_sensitivity(outputs, labels)
        self.valid_specificity(outputs, labels)
        self.valid_accuracy(outputs, labels)
        self.valid_f1score(outputs, labels)
        self.valid_meaniou(outputs, labels)
        self.log_dict(
            {
                "valid_loss": valid_loss,
                "valid_sensitivity": self.valid_sensitivity,
                "valid_specificity": self.valid_specificity,
                "valid_accuracy": self.valid_accuracy,
                "valid_f1score": self.valid_f1score,
                "valid_meaniou": self.valid_meaniou,
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        return valid_loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        test_loss = self.loss_fn(outputs, labels)
        self.test_sensitivity(outputs, labels)
        self.test_specificity(outputs, labels)
        self.test_accuracy(outputs, labels)
        self.test_f1score(outputs, labels)
        self.test_meaniou(outputs, labels)
        self.log_dict(
            {
                "test_loss": test_loss,
                "test_sensitivity": self.test_sensitivity,
                "test_specificity": self.test_specificity,
                "test_accuracy": self.test_accuracy,
                "test_f1score": self.test_f1score,
                "test_meaniou": self.test_meaniou,
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return self.optimizer
