import lightning as L
import torchmetrics
from torch import optim, nn, utils, Tensor


class CimatModule(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # Training metrics
        self.train_sensitivity = torchmetrics.Recall("binary")
        self.train_specificity = torchmetrics.Specificity("binary")
        self.train_accuracy = torchmetrics.Accuracy("binary")
        self.train_f1score = torchmetrics.F1Score("binary")
        self.train_meaniou = torchmetrics.JaccardIndex(task="binary")
        # Validation metrics
        self.valid_sensitivity = torchmetrics.Recall("binary")
        self.valid_specificity = torchmetrics.Specificity("binary")
        self.valid_accuracy = torchmetrics.Accuracy("binary")
        self.valid_f1score = torchmetrics.F1Score("binary")
        self.valid_meaniou = torchmetrics.JaccardIndex(task="binary")
        # Test metrics
        self.test_sensitivity = torchmetrics.Recall("binary")
        self.test_specificity = torchmetrics.Specificity("binary")
        self.test_accuracy = torchmetrics.Accuracy("binary")
        self.test_f1score = torchmetrics.F1Score("binary")
        self.test_meaniou = torchmetrics.JaccardIndex(task="binary")

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        train_loss = self.loss_fn(preds, labels)
        self.train_sensitivity(preds, labels)
        self.train_specificity(preds, labels)
        self.train_accuracy(preds, labels)
        self.train_f1score(preds, labels)
        self.train_meaniou(preds, labels)
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
        preds = self.model(inputs)
        valid_loss = self.loss_fn(preds, labels)
        self.valid_sensitivity(preds, labels)
        self.valid_specificity(preds, labels)
        self.valid_accuracy(preds, labels)
        self.valid_f1score(preds, labels)
        self.valid_meaniou(preds, labels)
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
        preds = self.model(inputs)
        test_loss = self.loss_fn(preds, labels)
        self.test_sensitivity(preds, labels)
        self.test_specificity(preds, labels)
        self.test_accuracy(preds, labels)
        self.test_f1score(preds, labels)
        self.test_meaniou(preds, labels)
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

    def configure_optimizers(self):
        return self.optimizer
