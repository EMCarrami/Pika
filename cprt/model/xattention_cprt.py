from typing import Optional, Tuple, cast

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.model.cprt_model import BaseCPrtModel
from cprt.model.helper_modules import CPrtCrossAttentionLayer


class CrossAttentionCPrt(BaseCPrtModel):
    """Flamingo-style CPrt model with cross-attention in all layers."""

    def __init__(
        self,
        language_model: str,
        protein_model: str,
        protein_layer_to_use: int = -1,
        perceiver_latent_size: int = 20,
        num_perceiver_layers: int = 1,
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize language and protein encoders."""
        super(CrossAttentionCPrt, self).__init__(language_model, protein_model, protein_layer_to_use)
        # add cross-attention layers
        protein_emb_size = cast(int, self.esm.embed_tokens.embedding_dim)
        protein_num_heads = self.esm.layers[0].self_attn.num_heads
        cross_attention_block = nn.ModuleList(
            [
                CPrtCrossAttentionLayer(
                    protein_emb_size,
                    decoder,
                    protein_num_heads,
                    perceiver_latent_size,
                    num_perceiver_layers,
                    enable_gradient_checkpointing,
                )
                for decoder in self.cprt_llm.transformer.h
            ]
        )
        self.cprt_llm.transformer.h = cross_attention_block

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
        """Use protein embeddings as encoder_hidden_states for cross attention."""
        with torch.no_grad():
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

    @torch.no_grad()
    def generate(self, protein_ids: Tensor, info_ids: Tensor, max_length: int = 50) -> Tensor:
        """Generate using input_ids and protein_embeddings as encoder_hidden_states."""
        self.eval()
        protein_embeddings = self.esm(protein_ids)
        preds: Tensor = self.cprt_llm.generate(
            input_ids=info_ids, encoder_hidden_states=protein_embeddings, use_cache=False, max_length=max_length
        )
        return preds
