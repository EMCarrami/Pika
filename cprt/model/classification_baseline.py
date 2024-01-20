import torch
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import F1Score, MeanAbsoluteError

from cprt.data.classification_datamodule import ClassificationData
from cprt.model.helper_modules import Perceiver, Squeeze, TruncatedESM2


class ProteinClassificationModel(LightningModule):  # type: ignore[misc]
    """Protein Classification Baseline Model."""

    criterion: nn.Module
    classifier: nn.Module

    def __init__(
        self,
        protein_model: str,
        classifier: str,
        num_classes: int,
        protein_layer_to_use: int = -1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize language and protein encoders."""
        super(ProteinClassificationModel, self).__init__()
        self.save_hyperparameters()

        self.protein_model = protein_model
        self.protein_layer_to_use = protein_layer_to_use
        self.lr = lr
        self.weight_decay = weight_decay

        if num_classes > 1:
            self.criterion = nn.CrossEntropyLoss()
            for s in ["train", "val", "test"]:
                setattr(self, f"{s}_metric", F1Score(task="multiclass", num_classes=num_classes))
        else:
            self.criterion = nn.MSELoss()
            for s in ["train", "val", "test"]:
                setattr(self, f"{s}_metric", MeanAbsoluteError())

        esm, _ = torch.hub.load("facebookresearch/esm:main", self.protein_model)  # type: ignore[no-untyped-call]
        self.esm = TruncatedESM2(esm, self.protein_layer_to_use)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

        emb_dim = self.esm.embedding_dim
        num_heads = self.esm.num_heads

        if classifier == "linear":
            self.preprocess = lambda x: x[:, 0]
            self.classifier = nn.Linear(emb_dim, num_classes)
        elif classifier == "mlp":
            self.preprocess = lambda x: x[:, 0]
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.GELU(),
                nn.Linear(emb_dim // 2, emb_dim // 4),
                nn.GELU(),
                nn.Linear(emb_dim // 4, num_classes),
            )
        elif classifier == "perceiver":
            self.preprocess = lambda x: x
            self.classifier = nn.Sequential(
                Perceiver(emb_dim, latent_size=1, output_dim=emb_dim, num_heads=num_heads, num_layers=1, dropout=0),
                Squeeze(1),
                nn.Linear(emb_dim, emb_dim // 2),
                nn.GELU(),
                nn.Linear(emb_dim // 2, num_classes),
            )

    def forward(self, input_ids: Tensor) -> Tensor:
        with torch.no_grad():
            embs = self.esm(input_ids)
        logits: Tensor = self.classifier(self.preprocess(embs))  # type: ignore[no-untyped-call]
        return logits

    def general_step(self, batch: ClassificationData, mode: str) -> Tensor:
        preds = self(batch.protein_ids).squeeze(-1)
        loss: Tensor = self.criterion(preds, batch.labels)
        getattr(self, f"{mode}_metric").update(preds, batch.labels)
        self.log(f"loss/{mode}_loss", loss.item())
        return loss

    def training_step(self, batch: ClassificationData) -> Tensor:
        return self.general_step(batch, "train")

    def validation_step(self, batch: ClassificationData) -> None:
        self.general_step(batch, "val")

    def test_step(self, batch: ClassificationData) -> None:
        self.general_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log("metrics/val_metric", self.val_metric.compute())
        self.val_metric.reset()
        self.log("metrics/train_metric", self.train_metric.compute())
        self.train_metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
