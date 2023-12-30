import gc
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from torchmetrics.text import Perplexity, ROUGEScore
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.data.cprt_datamodule import CPrtData, CPrtMetricData
from cprt.metrics.biochem_metrics import BiochemMetrics
from cprt.model.adapted_phi_models import PhiCPrt
from cprt.model.helper_modules import (
    CPrtCrossAttentionLayer,
    CPrtEncoderSkipLayer,
    CPrtSoftPromptLayer,
    GPT2CPrt,
    TruncatedESM2,
)


class CPrtModel(LightningModule):  # type: ignore[misc]
    """Abstract class of Cprt Lightning model."""

    MultiModalLayer: Type[CPrtCrossAttentionLayer] | Type[CPrtSoftPromptLayer]

    def __init__(
        self,
        language_model: str,
        protein_model: str,
        multimodal_strategy: Literal["cross-attention", "soft-prompt"],
        protein_layer_to_use: int = -1,
        perceiver_latent_size: int = 20,
        num_perceiver_layers: int = 1,
        multimodal_layers: List[int] | Literal["all"] = "all",
        enable_gradient_checkpointing: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        """Initialize language and protein encoders."""
        super(CPrtModel, self).__init__()
        self.save_hyperparameters()
        assert multimodal_strategy in ["cross-attention", "soft-prompt"]
        self.multimodal_strategy = multimodal_strategy
        if multimodal_strategy == "soft-prompt":
            assert multimodal_layers == [
                0
            ], "For soft-prompting only first decoder can be made multimodal. Set multimodal_layers==[0]"
            self.MultiModalLayer = CPrtSoftPromptLayer
        elif multimodal_strategy == "cross-attention":
            self.MultiModalLayer = CPrtCrossAttentionLayer
        else:
            raise ValueError("only cross-attention and soft-prompt multimodal strategies are supported.")
        self.language_model = language_model
        self.protein_model = protein_model
        self.protein_layer_to_use = protein_layer_to_use
        self.perceiver_latent_size = perceiver_latent_size
        self.num_perceiver_layers = num_perceiver_layers
        self.multimodal_layers = multimodal_layers
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.lr = lr
        self.weight_decay = weight_decay

        self.configure_protein_model()
        self.configure_language_model()
        self.configure_metrics()

    def configure_protein_model(self) -> None:
        esm, _ = torch.hub.load("facebookresearch/esm:main", self.protein_model)  # type: ignore[no-untyped-call]
        self.esm = TruncatedESM2(esm, self.protein_layer_to_use)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

    def configure_language_model(self) -> None:
        """Configure cprt_llm model and add cross-attention/soft-prompt/skip layers to the decoder."""
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        if "gpt2" in self.language_model:
            self.cprt_llm = GPT2CPrt.from_pretrained(self.language_model)
        elif "phi" in self.language_model:
            self.cprt_llm = PhiCPrt.from_pretrained(self.language_model)
        else:
            raise ValueError(f"only gpt2 and microsoft/phi models are supported. {self.language_model} was given")
        assert (
            self.cprt_llm.transformer.gradient_checkpointing is False
        ), "gradient_checkpointing in LLMs not supported as the order of args to the input cannot be guaranteed."
        for param in self.cprt_llm.parameters():
            param.requires_grad = False

        protein_emb_dim = self.esm.embedding_dim
        protein_num_heads = self.esm.num_heads
        text_emb_dim = self.cprt_llm.config.n_embd
        dropout = self.cprt_llm.config.attn_pdrop
        num_decoder_heads = self.cprt_llm.config.n_head
        decoder_layers = self.cprt_llm.transformer.h
        n_llm_layers = len(decoder_layers)
        # assign which layers to make multimodal
        if self.multimodal_layers == "all":
            self.multimodal_layers = list(range(n_llm_layers))
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
                else CPrtEncoderSkipLayer(decoder)
                for layer_id, decoder in enumerate(decoder_layers)
            ]
        )
        self.cprt_llm.transformer.h = multimodal_block

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
        if self.multimodal_strategy == "soft-prompt":
            # Extend info_ids, labels and mask to accommodate for protein latents
            prompt_size = (info_ids.size(0), self.perceiver_latent_size)
            info_ids = torch.concat([info_ids.new_full(size=prompt_size, fill_value=0), info_ids], dim=1)
            if labels is not None:
                labels = torch.concat([labels.new_full(size=prompt_size, fill_value=-100), labels], dim=1)
            attention_mask = torch.concat(
                [attention_mask.new_full(size=prompt_size, fill_value=1), attention_mask], dim=1
            )
        out = self.cprt_llm(
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
        if self.multimodal_strategy == "soft-prompt":
            out["logits"] = out["logits"][:, self.perceiver_latent_size :]
        return out

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
            if batch_idx == 0 and not self.trainer.sanity_checking:
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

    @torch.no_grad()
    def generate(
        self, protein_ids: Tensor, info_ids: Tensor, generation_length: int = 20, keep_prompt: bool = False
    ) -> List[str]:
        """Generate using input_ids and protein_embeddings as encoder_hidden_states."""
        self.eval()
        protein_embeddings = self.esm(protein_ids)
        if self.multimodal_strategy == "soft-prompt":
            info_ids = torch.concat(
                [info_ids.new_full(size=(info_ids.size(0), self.perceiver_latent_size), fill_value=0), info_ids], dim=1
            )
        out = []
        if torch.any(info_ids == self.text_tokenizer.eos_token_id):
            for question, protein in zip(info_ids, protein_embeddings):
                mask = question == self.text_tokenizer.eos_token_id
                prompt_len = cast(int, torch.where(mask)[0][0] if mask.any() else len(question))
                pos_offset = 0 if keep_prompt else prompt_len
                preds = self.cprt_llm.generate(
                    input_ids=question[:prompt_len].unsqueeze(0),
                    encoder_hidden_states=protein.unsqueeze(0),
                    use_cache=False,
                    max_length=generation_length + prompt_len,
                )
                out.append(self.text_tokenizer.decode(preds[0, pos_offset:].cpu(), skip_special_tokens=True))
        else:
            prompt_len = info_ids.size(1)
            pos_offset = 0 if keep_prompt else prompt_len
            preds = self.cprt_llm.generate(
                input_ids=info_ids,
                encoder_hidden_states=protein_embeddings,
                use_cache=False,
                max_length=generation_length + prompt_len,
            )
            for pred in preds:
                out.append(self.text_tokenizer.decode(pred[pos_offset:].cpu(), skip_special_tokens=True))
        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
        self.val_rouge_scores = ROUGEScore()
        self.val_biochem = BiochemMetrics()
        self.text_table = wandb.Table(  # type: ignore[no-untyped-call]
            columns=["global_step", "expected_answer", "generated_text"]
        )
