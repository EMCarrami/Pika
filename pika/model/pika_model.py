import gc
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, cast

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchmetrics.text import Perplexity, ROUGEScore
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from pika.datamodule.pika_datamodule import PikaData, PikaMetricData
from pika.metrics.biochem_lite_metrics import BiochemLiteMetrics
from pika.model.helper_modules import TruncatedESM2
from pika.model.pika_modules import (
    CrossPikaAttentionLayer,
    GPT2ForPika,
    PhiForPika,
    PikaEncoderSkipLayer,
    SelfPikaLayer,
)


class PikaModel(LightningModule):  # type: ignore[misc]
    """Pika Lightning model."""

    MultiModalLayer: Type[CrossPikaAttentionLayer] | Type[SelfPikaLayer]

    def __init__(
        self,
        language_model: str,
        protein_model: str,
        multimodal_strategy: Literal["cross-pika", "self-pika"],
        protein_layer_to_use: int = -1,
        perceiver_latent_size: int = 20,
        num_perceiver_layers: int = 1,
        multimodal_layers: List[int] | Literal["all"] = "all",
        enable_gradient_checkpointing: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        schedulers: List[str] | None = None,
    ) -> None:
        """Initialize language and protein encoders."""
        super(PikaModel, self).__init__()
        self.save_hyperparameters()
        assert multimodal_strategy in ["cross-pika", "self-pika"]
        self.multimodal_strategy = multimodal_strategy
        if multimodal_strategy == "self-pika":
            assert multimodal_layers == [
                0
            ], "For self-pika only first decoder can be made multimodal. Set multimodal_layers==[0]"
            self.MultiModalLayer = SelfPikaLayer
        elif multimodal_strategy == "cross-pika":
            self.MultiModalLayer = CrossPikaAttentionLayer
        else:
            raise ValueError("only cross-pika and self-pika multimodal strategies are supported.")
        self.language_model = language_model
        self.protein_model = protein_model
        self.protein_layer_to_use = protein_layer_to_use
        self.perceiver_latent_size = perceiver_latent_size
        self.num_perceiver_layers = num_perceiver_layers
        self.multimodal_layers = multimodal_layers
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.lr = lr
        self.weight_decay = weight_decay
        self.schedulers = schedulers

        self.configure_protein_model()
        self.configure_language_model()
        self.configure_metrics()

    def configure_protein_model(self) -> None:
        """Configure ESM2 model."""
        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.protein_model}")
        esm, _ = torch.hub.load("facebookresearch/esm:main", self.protein_model)  # type: ignore[no-untyped-call]
        self.esm = TruncatedESM2(esm, self.protein_layer_to_use)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

    def configure_language_model(self) -> None:
        """Configure pika_llm model and add cross-attention/soft-prompt/skip layers to the decoder."""
        if "gpt2" in self.language_model:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.language_model, use_fast=False)
            self.pika_llm = GPT2ForPika.from_pretrained(self.language_model)
        elif "phi" in self.language_model:
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.language_model, use_fast=False, revision="7e10f3ea09c0ebd373aebc73bc6e6ca58204628d"
            )
            self.pika_llm = PhiForPika.from_pretrained(
                self.language_model,
                flash_attn=True,
                flash_rotary=True,
                fused_dense=True,
                revision="7e10f3ea09c0ebd373aebc73bc6e6ca58204628d",
            )
        else:
            raise ValueError(f"only gpt2 and microsoft/phi models are supported. {self.language_model} was given")
        assert (
            self.pika_llm.transformer.gradient_checkpointing is False
        ), "gradient_checkpointing in LLMs not supported as the order of args to the input cannot be guaranteed."

        for param in self.pika_llm.parameters():
            param.requires_grad = False

        protein_emb_dim = self.esm.embedding_dim
        protein_num_heads = self.esm.num_heads
        text_emb_dim = self.pika_llm.config.n_embd
        dropout = self.pika_llm.config.attn_pdrop
        num_decoder_heads = self.pika_llm.config.n_head
        decoder_layers = self.pika_llm.transformer.h
        n_llm_layers = len(decoder_layers)
        # assign which layers to make multimodal
        if isinstance(self.multimodal_layers, str) and self.multimodal_layers.startswith("all"):
            eliminate = self.multimodal_layers.split("-")
            eliminate_list = []
            if len(eliminate) > 1:
                eliminate_list = [int(i) for i in eliminate[1].split(",")]
            self.multimodal_layers = [i for i in range(n_llm_layers) if i not in eliminate_list]
        assert (
            isinstance(self.multimodal_layers, list)
            and all([isinstance(i, int) for i in self.multimodal_layers])
            and max(self.multimodal_layers) < n_llm_layers
        ), "multimodal_layers must be all or list[int]"
        self.multimodal_layers = [i if i >= 0 else n_llm_layers + i for i in self.multimodal_layers]
        # make decoder layers multimodal or skip-encoder
        multimodal_block = torch.nn.ModuleList(
            [
                self.MultiModalLayer(
                    protein_emb_dim,
                    text_emb_dim,
                    decoder,
                    num_decoder_heads,
                    protein_num_heads,
                    self.perceiver_latent_size,
                    self.num_perceiver_layers,
                    self.enable_gradient_checkpointing,
                    dropout,
                )
                if layer_id in self.multimodal_layers
                else PikaEncoderSkipLayer(decoder)
                for layer_id, decoder in enumerate(decoder_layers)
            ]
        )
        self.pika_llm.transformer.h = multimodal_block

    def forward(
        self,
        protein_ids: Tensor,
        info_ids: Tensor,
        attention_mask: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        """Add protein latents as word embedding to the beginning of the sentence."""
        with torch.no_grad():
            protein_embeddings = self.esm(protein_ids)
        if self.multimodal_strategy == "self-pika":
            # Extend info_ids, labels and mask to accommodate for protein latents
            prompt_size = (info_ids.size(0), self.perceiver_latent_size)
            info_ids = torch.concat([info_ids.new_full(size=prompt_size, fill_value=0), info_ids], dim=1)
            if labels is not None:
                labels = torch.concat([labels.new_full(size=prompt_size, fill_value=-100), labels], dim=1)
            attention_mask = torch.concat(
                [attention_mask.new_full(size=prompt_size, fill_value=1), attention_mask], dim=1
            )
        out = self.pika_llm(
            input_ids=info_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            encoder_hidden_states=protein_embeddings,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # remove logits of the protein latents
        if self.multimodal_strategy == "self-pika":
            out["logits"] = out["logits"][:, self.perceiver_latent_size :]
        return out

    def training_step(self, batch: PikaData, batch_idx: int) -> Tensor:
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

    def validation_step(self, batch: PikaData | PikaMetricData, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Take a val step."""
        if dataloader_idx == 0:
            assert isinstance(batch, PikaData)
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
        else:
            assert isinstance(batch, PikaMetricData)
            out = self.get_response(
                protein_ids=batch.protein, info_ids=batch.question, generation_length=20, keep_prompt=False
            )
            self.val_biochem.update(out, batch.expected_value, batch.metric_name)
            self.log_example_outputs(out, batch)
        torch.cuda.empty_cache()
        gc.collect()

    def on_test_start(self) -> None:
        """Create wandb table for Biochem-ReAct metrics."""
        self.test_results: List[List[str]] = []

    def test_step(self, batch: PikaMetricData, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform test on Biochem-ReAct metrics."""
        out = self.get_response(
            protein_ids=batch.protein, info_ids=batch.question, generation_length=60, keep_prompt=False
        )
        for n, e, p in zip(batch.metric_name, batch.expected_value, out):
            uid, s = n.split(": ")
            self.test_results.append([uid, s, e, p])
        torch.cuda.empty_cache()

    def on_test_end(self) -> None:
        """Log test outcomes to wandb."""
        if isinstance(self.logger, WandbLogger):
            test_table = wandb.Table(  # type: ignore[no-untyped-call]
                columns=["uniprot_id", "subject", "expected_answer", "generated_response"]
            )
            for v in self.test_results:
                test_table.add_data(*v)  # type: ignore[no-untyped-call]
            wandb.log({"Biochem-ReAct_results": test_table})

    def log_example_outputs(self, output_text: List[str], batch: PikaMetricData) -> None:
        """
        Log example generated responses.

        Logs up to two examples from each question type by checking the first examples in each batch
        """
        name = batch.metric_name[0]
        for num in [1, 2]:
            if f"{name}_{self.global_step}_{num}" not in self.val_example_outputs:
                self.val_example_outputs[f"{name}_{self.global_step}_{num}"] = {
                    "global_step": self.global_step,
                    "question": self.text_tokenizer.decode(batch.question[0], skip_special_tokens=True),
                    "expected_answer": str(batch.expected_value[0]),
                    "generated_response": output_text[0],
                }
                break

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log("metrics/val_perplexity", self.val_perplexity.compute())
        self.val_perplexity.reset()
        rouge_scores: Dict[str, Tensor] = self.val_rouge_scores.compute()
        self.log_dict({f"metrics/val_{k}": v.mean() for k, v in rouge_scores.items()})
        self.val_rouge_scores.reset()
        biochem_scores = self.val_biochem.compute()
        self.log_dict({f"Biochem-Lite/val_{k}": v for k, v in biochem_scores.items()})
        self.val_biochem.reset()

    def on_train_epoch_end(self) -> None:
        """Log generation examples table."""
        self.log_wandb_table()

    @rank_zero_only  # type: ignore[misc]
    def log_wandb_table(self) -> None:
        """Log wandb table of example outputs."""
        if isinstance(self.logger, WandbLogger):
            text_table = wandb.Table(  # type: ignore[no-untyped-call]
                columns=["global_step", "question", "expected_answer", "generated_response"]
            )
            for v in self.val_example_outputs.values():
                text_table.add_data(*v.values())  # type: ignore[no-untyped-call]
            wandb.log({f"val_generation_{self.current_epoch}": text_table})

    def configure_optimizers(self) -> Optimizer | Tuple[List[Optimizer], List[Dict[str, Any]]]:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.schedulers is None:
            return optimizer
        else:
            total_samples = len(self.trainer.datamodule.train_dataloader().dataset)
            batch_size = self.trainer.datamodule.train_batch_size
            accumulate_grad_batches = self.trainer.accumulate_grad_batches
            total_steps_per_epoch = (total_samples // batch_size) // accumulate_grad_batches
            warmup_steps = int(total_steps_per_epoch * 0.05)
            schedulers = []
            if "warmup" in self.schedulers:
                warmup_scheduler = LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))
                schedulers.append({"scheduler": warmup_scheduler, "interval": "step"})
            if "cosine" in self.schedulers:
                anneal_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps_per_epoch)
                schedulers.append({"scheduler": anneal_scheduler, "interval": "step"})
            return [optimizer], schedulers

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
        self.val_rouge_scores = ROUGEScore()
        self.val_biochem = BiochemLiteMetrics()
        self.val_example_outputs: Dict[str, Dict[str, str]] = OrderedDict()

    @torch.no_grad()
    def get_response(
        self, protein_ids: Tensor, info_ids: Tensor, generation_length: int = 20, keep_prompt: bool = False
    ) -> List[str]:
        """Generate text using input_ids and protein_embeddings as encoder_hidden_states."""
        self.eval()
        protein_embeddings = self.esm(protein_ids)
        if self.multimodal_strategy == "self-pika":
            info_ids = torch.concat(
                [info_ids.new_full(size=(info_ids.size(0), self.perceiver_latent_size), fill_value=0), info_ids], dim=1
            )
        out = []
        if torch.any(info_ids == self.text_tokenizer.eos_token_id):
            for question, protein in zip(info_ids, protein_embeddings):
                mask = question == self.text_tokenizer.eos_token_id
                prompt_len = cast(int, torch.where(mask)[0][0] if mask.any() else len(question))
                pos_offset = 0 if keep_prompt else prompt_len
                preds = self.pika_llm.generate(
                    input_ids=question[:prompt_len].unsqueeze(0),
                    encoder_hidden_states=protein.unsqueeze(0),
                    use_cache=False,
                    max_length=generation_length + prompt_len,
                )
                # fix for phi-2 tokenizer
                txt = self.text_tokenizer.decode(preds[0, pos_offset:].cpu(), skip_special_tokens=False)
                out.append(txt.split("<|endoftext|>")[0])
        else:
            prompt_len = info_ids.size(1)
            pos_offset = 0 if keep_prompt else prompt_len
            preds = self.pika_llm.generate(
                input_ids=info_ids,
                encoder_hidden_states=protein_embeddings,
                use_cache=False,
                max_length=generation_length + prompt_len,
            )
            for pred in preds:
                # fix for phi-2 tokenizer
                txt = self.text_tokenizer.decode(pred[pos_offset:].cpu(), skip_special_tokens=False)
                out.append(txt.split("<|endoftext|>")[0])
        return out

    @torch.no_grad()
    def enquire(
        self,
        proteins: List[str],
        question: str,
        generation_length: int = 20,
        keep_prompt: bool = False,
        placeholder: str | None = None,
    ) -> List[str]:
        """Generate answer to the question for a given protein sequence."""
        if placeholder is None:
            if hasattr(self.trainer, "datamodule"):
                placeholder = self.trainer.datamodule.sequence_placeholder
                logger.info(f"using datamodules placeholeder {placeholder}")
            else:
                placeholder = "<protein sequence placeholder> "
                logger.info(
                    f"using default placeholeder {placeholder}. "
                    f"Ensure this is intended. If not please provide a placeholder."
                )
        question = f"{self.sequence_placeholder}{question}"
        return self.get_response(
            protein_ids=self.protein_tokenizer(proteins, padding=True, return_tensors="pt")["input_ids"],
            info_ids=self.text_tokenizer([question] * len(proteins), return_tensors="pt")["input_ids"],
            generation_length=generation_length,
            keep_prompt=keep_prompt,
        )
