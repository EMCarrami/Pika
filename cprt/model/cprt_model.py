import gc
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import LongTensor, Tensor
from torchmetrics.text import Perplexity, ROUGEScore
from transformers import GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.data.cprt_datamodule import CPrtData, CPrtMetricData
from cprt.metrics.biochem_metrics import BiochemMetrics
from cprt.model.helper_modules import GPT2CPrt, TruncatedESM2


class BaseCPrtModel(LightningModule, ABC):  # type: ignore[misc]
    """Abstract class of Cprt Lightning model."""

    def __init__(
        self,
        language_model: str,
        protein_model: str,
        protein_layer_to_use: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        """Initialize language and protein encoders."""
        super(BaseCPrtModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        esm, _ = torch.hub.load("facebookresearch/esm:main", protein_model)  # type: ignore[no-untyped-call]
        self.esm = TruncatedESM2(esm, protein_layer_to_use)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

        self.cprt_llm = GPT2CPrt.from_pretrained(language_model)
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(language_model)
        for param in self.cprt_llm.parameters():
            param.requires_grad = False

        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
        self.val_rouge_scores = ROUGEScore()
        self.val_biochem = BiochemMetrics()

        self.text_table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=["global_step", "expected_answer", "generated_text"]
        )

    @abstractmethod
    def forward(
        self,
        protein_ids: Tensor,
        info_ids: Tensor,
        attention_mask: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        labels: Optional[LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        """Implement GPT2 compatible forward."""
        raise NotImplementedError("Implement a GPT2 compatible forward.")

    def training_step(self, batch: CPrtData, batch_idx: int) -> Tensor:
        """Take a train step."""
        out = self(protein_ids=batch.protein, info_ids=batch.info, attention_mask=batch.info_mask, labels=batch.labels)
        loss: Tensor = out["loss"]
        self.log("loss/train_loss", loss.item(), prog_bar=True)

        self.train_perplexity.update(out["logits"].detach()[:, :-1].float(), batch.labels[:, 1:])
        if batch_idx % self.trainer.val_check_interval == 0 and batch_idx != 0:
            self.log("metrics/train_perplexity", self.train_perplexity.compute())
            self.train_perplexity.reset()

        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch: CPrtData | CPrtMetricData, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Take a val step."""
        if dataloader_idx == 0:
            assert isinstance(batch, CPrtData)
            out = self(
                protein_ids=batch.protein,
                info_ids=batch.info,
                attention_mask=batch.info_mask,
                labels=batch.labels,
            )
            loss: Tensor = out["loss"]
            self.log("loss/val_loss", loss.item(), prog_bar=True, add_dataloader_idx=False)
            self.val_perplexity.update(out["logits"][:, :-1].float(), batch.labels[:, 1:])
            # generation metrics
            input_text = self.text_tokenizer.batch_decode(batch.info, skip_special_tokens=True)
            generated_text = self.text_tokenizer.batch_decode(
                torch.argmax(out["logits"], dim=-1), skip_special_tokens=True
            )
            self.val_rouge_scores.update(generated_text, input_text)
            if batch_idx == 0:
                self.log_example_outputs(input_text[-4:], batch.protein[-4:])
        else:
            assert isinstance(batch, CPrtMetricData)
            out = self.generate(protein_ids=batch.protein, info_ids=batch.info, generation_length=20, keep_prompt=False)
            self.val_biochem.update(out, batch.expected_value, batch.metric_name)
        torch.cuda.empty_cache()
        gc.collect()

    def log_example_outputs(self, input_text: List[str], protein: Tensor) -> None:
        """Log example generated responses."""
        for in_txt, protein in zip(input_text, protein):
            if "?" in in_txt:
                question = in_txt.split("?")[0]
                response = self.generate(
                    protein_ids=protein.unsqueeze(0),
                    info_ids=self.text_tokenizer(f"{question}? ", return_tensors="pt")["input_ids"].to(self.device),
                    generation_length=25,
                    keep_prompt=True,
                )
                self.text_table.add_data(self.trainer.global_step, in_txt, response)  # type: ignore[no-untyped-call]

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log("metrics/val_perplexity", self.val_perplexity.compute())
        self.val_perplexity.reset()
        rouge_scores: Dict[str, Tensor] = self.val_rouge_scores.compute()
        self.log_dict({f"metrics/val_{k}": v.mean() for k, v in rouge_scores.items()})
        self.val_rouge_scores.reset()
        biochem_scores = self.val_biochem.compute()
        self.log_dict({f"biochem/val_{k}": v for k, v in biochem_scores.items()})
        self.val_biochem.reset()

    def on_train_end(self) -> None:
        """Log generation examples table."""
        self.log_wandb_table()

    @rank_zero_only  # type: ignore[misc]
    def log_wandb_table(self) -> None:
        """Log wandb table of example outputs."""
        wandb.log({"val_generation": self.text_table})

    @abstractmethod
    def generate(
        self, protein_ids: Tensor, info_ids: Tensor, generation_length: int = 20, keep_prompt: bool = False
    ) -> List[str]:
        """Implement generate method that returns a list of generated texts."""
        raise NotImplementedError("generate method for the model not implemented.")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
