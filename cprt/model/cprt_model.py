import gc
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import FloatTensor, LongTensor, Tensor
from torchmetrics.text import Perplexity, ROUGEScore
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.data.cprt_datamodule import CprtData
from cprt.model.helper_modules import GPT2CPrt, TruncatedESM2


class BaseCPrtModel(LightningModule, ABC):  # type: ignore[misc]
    """Abstract class of Cprt Lightning model."""

    def __init__(self, language_model: str, protein_model: str, protein_layer_to_use: int = -1) -> None:
        """Initialize language and protein encoders."""
        super().__init__()

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

        self.text_table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=["global_step", "expected_answer", "generated_text"]
        )

    @abstractmethod
    def forward(
        self,
        protein_ids: Tensor,
        info_ids: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        labels: Optional[LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        """Implement GPT2 compatible forward."""
        raise NotImplementedError("Implement GPT2 compatible forward.")

    def training_step(self, batch: CprtData, batch_idx: int) -> Tensor:
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

    def validation_step(self, batch: CprtData, batch_idx: int) -> None:
        """Take a val step."""
        out = self(
            protein_ids=batch.protein,
            info_ids=batch.info,
            attention_mask=batch.info_mask,
            labels=batch.labels,
        )
        loss: Tensor = out["loss"]
        self.log("loss/val_loss", loss.item(), prog_bar=True)
        self.val_perplexity.update(out["logits"][:, :-1].float(), batch.labels[:, 1:])
        # generation metrics
        input_text = self.text_tokenizer.batch_decode(batch.info, skip_special_tokens=True)
        generated_text = self.text_tokenizer.batch_decode(torch.argmax(out["logits"], dim=-1), skip_special_tokens=True)
        self.val_rouge_scores.update(generated_text, input_text)
        if batch_idx == 0:
            self.log_example_outputs(input_text[:4], batch.protein[:4])
        torch.cuda.empty_cache()
        gc.collect()

    def log_example_outputs(self, input_text: List[str], protein: Tensor) -> None:
        """Log example generated responses."""
        for in_txt, protein in zip(input_text, protein):
            if "?" in in_txt:
                question = in_txt.split("?")[0]
                preds = self.generate(
                    protein_ids=protein.unsqueeze(0),
                    info_ids=self.text_tokenizer(f"{question}? ", return_tensors="pt")["input_ids"].to(self.device),
                    max_length=50,
                )
                response = self.text_tokenizer.decode(preds[0].cpu())
                self.text_table.add_data(self.trainer.global_step, in_txt, response)  # type: ignore[no-untyped-call]

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log("metrics/val_perplexity", self.val_perplexity.compute())
        self.val_perplexity.reset()
        rouge_scores: Dict[str, Tensor] = self.val_rouge_scores.compute()
        self.log_dict({f"metrics/val_{k}": v.mean() for k, v in rouge_scores.items()})
        self.val_rouge_scores.reset()

    def on_fit_end(self) -> None:
        """Log generation examples table."""
        self.log_wandb_table()
        self.log_biochem_metrics()

    @rank_zero_only  # type: ignore[misc]
    def log_wandb_table(self) -> None:
        """Log wandb table of example outputs."""
        wandb.log({"val_generation": self.text_table})

    @rank_zero_only  # type: ignore[misc]
    def log_biochem_metrics(self) -> None:
        """Compute and log biochemical metrics on validation set."""
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.load_state_dict(torch.load(best_model_path)["state_dict"])

        locations = ["membrane", "nucleus", "mitochondri"]
        question = "What is the subcellular location of this protein?"
        loader = self.trainer.val_dataloaders

        seqs, loc_info = defaultdict(list), defaultdict(list)
        for batch in loader:
            for info, seq in zip(batch.info, batch.protein):
                info_text = self.text_tokenizer.decode(info, skip_special_tokens=True)
                if "where" in info_text.split("?")[0].lower() or "location" in info_text.split("?")[0].lower():
                    for lc in locations:
                        if lc in info_text:
                            seqs[lc].append(seq)
                            loc_info[lc].append(info_text)

        tokenized_question = self.text_tokenizer(
            [f"{self.trainer.datamodule.sequence_placeholder} {question}"], return_tensors="pt"
        )
        for lc in locations:
            print(f"predicting {lc}, {len(seqs[lc])}")
            correct = 0
            for seq, expected in tqdm(zip(seqs[lc], loc_info[lc]), total=len(seqs[lc])):
                with torch.no_grad():
                    preds = self.generate(
                        protein_ids=seq.unsqueeze(0),
                        info_ids=tokenized_question["input_ids"].to(self.device),
                        max_length=50,
                    )
                response = self.text_tokenizer.decode(preds[0].cpu())
                if lc in response:
                    correct += 1
            self.log(f"biochem/{lc}_accuracy", correct / len(seqs[lc]))

    @abstractmethod
    def generate(self, protein_ids: Tensor, info_ids: Tensor, max_length: int = 50) -> Tensor:
        """Implement generate method."""
        raise NotImplementedError("generate method for the model not implemented.")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer

    def on_keyboard_interrupt(self) -> None:
        print("Keyboard interrupt detected. Gracefully shutting down...")
        self.on_fit_end()
