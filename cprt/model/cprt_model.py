from typing import Any, Dict, Optional, Tuple, cast

import torch
from lightning import LightningModule
from torch import FloatTensor, LongTensor, Tensor, nn
from torchmetrics.text import BERTScore, BLEUScore, Perplexity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.data.cprt_datamodule import CprtData
from cprt.model.helper_modules import CrossAttentionDecoderLayer, TruncatedESM2


class Cprt(LightningModule):  # type: ignore[misc]
    """Class of Cprt Lightning model."""

    def __init__(
        self,
        language_model: str = "gpt2",
        protein_model: str = "esm2_t6_8M_UR50D",
        protein_layer_to_use: int = -1,
    ) -> None:
        """Initialize language and protein encoders."""
        super().__init__()

        esm, _ = torch.hub.load("facebookresearch/esm:main", protein_model)  # type: ignore[no-untyped-call]
        self.esm = TruncatedESM2(esm, protein_layer_to_use)
        for param in self.esm.parameters():
            param.requires_grad = False

        self.cprt_llm = GPT2LMHeadModel.from_pretrained(language_model)
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(language_model)
        for param in self.cprt_llm.parameters():
            param.requires_grad = False
        self._add_cross_attention_to_llm()
        self._modify_generation_input_to_llm()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
        self.val_bleu_score = BLEUScore()
        self.val_bert_scores = BERTScore()

    def _add_cross_attention_to_llm(self) -> None:
        """Add Cross-Attention layers to all decoder blocks."""
        protein_emb_size = cast(int, self.esm.embed_tokens.embedding_dim)
        protein_num_heads = self.esm.layers[0].self_attn.num_heads
        cross_attention_block = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(protein_emb_size, decoder, num_perceiver_heads=protein_num_heads)
                for decoder in self.cprt_llm.transformer.h
            ]
        )
        self.cprt_llm.transformer.h = cross_attention_block

    def _modify_generation_input_to_llm(self) -> None:
        """
        Update the cprt_llm prepare_inputs_for_generation method.

        TODO: Add details and explain why
        """
        original_method = self.cprt_llm.prepare_inputs_for_generation

        def updated_prepare_inputs_for_generation(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            """Add encoder_hidden_states to model_inputs of GPT2 generation."""
            model_inputs: Dict[str, Any] = original_method(*args, **kwargs)
            model_inputs["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
            return model_inputs

        self.cprt_llm.prepare_inputs_for_generation = updated_prepare_inputs_for_generation

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

        protein_embeddings = self.esm(protein_ids)

        return self.cprt_llm(
            input_ids=info_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=protein_embeddings,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: CprtData, batch_idx: int) -> Tensor:
        """Take a train step."""
        out = self(protein_ids=batch.protein, info_ids=batch.info, attention_mask=batch.info_mask)
        loss: Tensor = self.criterion(out["logits"].transpose(1, 2), batch.labels)
        self.log("loss/train_loss", loss.item(), prog_bar=True)
        self.train_perplexity.update(out["logits"].detach(), batch.labels)
        if batch_idx % self.trainer.val_check_interval == 0 and batch_idx != 0:
            self.log("metrics/train_perplexity", self.train_perplexity.compute(), on_step=True)
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
        loss: Tensor = self.criterion(out["logits"].transpose(1, 2), batch.labels)
        self.log("loss/val_loss", loss.item(), prog_bar=True)
        self.log("loss/val_gpt2_loss", out["loss"].item(), prog_bar=True)
        self.val_perplexity.update(out["logits"], batch.labels)
        input_text = self.text_tokenizer.batch_decode(batch.info, skip_special_tokens=True)
        generated_text = self.text_tokenizer.batch_decode(torch.argmax(out["logits"], dim=-1), skip_special_tokens=True)
        self.val_bleu_score.update(generated_text, input_text)
        # self.val_bert_scores.update(generated_text, input_text)
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        self.log("metrics/val_perplexity", self.val_perplexity.compute())
        self.val_perplexity.reset()
        self.log("metrics/val_bleu", self.val_bleu_score.compute())
        self.val_bleu_score.reset()
        # bert_score: Dict[str, Tensor] = self.val_bert_scores.compute()  # type: ignore[assignment]
        # self.log_dict({f"metrics/val_bert_{k}": v.mean() for k, v in bert_score.items()})
        # self.val_bert_scores.reset()
        for idx, layer in enumerate(self.cprt_llm.transformer.h):
            self.log(f"gates/layer_{idx}_attn_gate", layer.attn_gate.item())
            self.log(f"gates/layer_{idx}_ff_gate", layer.ff_gate.item())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer
