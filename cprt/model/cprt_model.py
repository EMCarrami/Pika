from typing import Any, Dict, Optional, Tuple, cast

import torch
from lightning import LightningModule
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers import GPT2LMHeadModel
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
        for param in self.cprt_llm.parameters():
            param.requires_grad = False
        self._add_cross_attention_to_llm()
        self._modify_generation_input_to_llm()

    def _add_cross_attention_to_llm(self) -> None:
        """Add Cross-Attention layers to all decoder blocks."""
        protein_emb_size = cast(int, self.esm.embed_tokens.embedding_dim)
        cross_attention_block = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(protein_emb_size, decoder)
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

        def updated_prepare_inputs_for_generation(
            *args: Any, **kwargs: Any
        ) -> Dict[str, Any]:
            """Add encoder_hidden_states to model_inputs of GPT2 generation."""
            model_inputs: Dict[str, Any] = original_method(*args, **kwargs)
            model_inputs["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
            return model_inputs

        self.cprt_llm.prepare_inputs_for_generation = (
            updated_prepare_inputs_for_generation
        )

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

    def general_step(self, batch: CprtData, mode: str) -> Tensor:
        """Take a general step."""
        out = self(
            protein_ids=batch.protein,
            info_ids=batch.info,
            attention_mask=batch.info_mask,
            labels=batch.labels,
        )
        loss: Tensor = out["loss"]
        self.log(f"loss/{mode}_loss", loss, prog_bar=True, on_step=True)
        torch.cuda.empty_cache()
        return loss

    def training_step(self, batch: CprtData, batch_idx: int) -> Tensor:
        """Take a train step."""
        for idx, layer in enumerate(self.cprt_llm.transformer.h):
            self.log(
                f"gates/layer_{idx}_attn_gate", layer.attn_gate.item(), on_step=True
            )
            self.log(f"gates/layer_{idx}_ff_gate", layer.ff_gate.item(), on_step=True)
        return self.general_step(batch, "train")

    def validation_step(self, batch: CprtData, batch_idx: int) -> None:
        """Take a val step."""
        self.general_step(batch, "val")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return {
            "optimizer": optimizer,
            "monitor": "loss/train_loss",
        }
