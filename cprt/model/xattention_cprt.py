from typing import List, Optional, Tuple, cast

import torch
from torch import LongTensor, Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.model.cprt_model import BaseCPrtModel
from cprt.model.helper_modules import CPrtCrossAttentionLayer, CPrtDummyCrossAttention


class CrossAttentionCPrt(BaseCPrtModel):
    """Flamingo-style CPrt model with cross-attention in all layers."""

    def __init__(
        self,
        language_model: str,
        protein_model: str,
        protein_layer_to_use: int = -1,
        perceiver_latent_size: int = 20,
        num_perceiver_layers: int = 1,
        layers_to_cross_attend: List[int] | None = None,
        enable_gradient_checkpointing: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        """Initialize language and protein encoders."""
        super(CrossAttentionCPrt, self).__init__(language_model, protein_model, protein_layer_to_use, lr, weight_decay)
        self.save_hyperparameters()
        self._set_layers_to_cross_attend(layers_to_cross_attend)
        self._add_cross_attention_layers_to_decoder(
            perceiver_latent_size, num_perceiver_layers, enable_gradient_checkpointing
        )

    def _add_cross_attention_layers_to_decoder(
        self,
        perceiver_latent_size: int,
        num_perceiver_layers: int,
        enable_gradient_checkpointing: bool,
    ) -> None:
        """Add cross-attention layers or dummy ones when cross-attention not needed."""
        protein_emb_dim = self.esm.embedding_dim
        protein_num_heads = self.esm.num_heads
        text_emb_dim = self.cprt_llm.config.n_embd
        dropout = self.cprt_llm.config.attn_pdrop
        num_decoder_heads = self.cprt_llm.config.n_head
        decoder_layers = self.cprt_llm.transformer.h

        cross_attention_block = nn.ModuleList(
            [
                CPrtCrossAttentionLayer(
                    protein_emb_dim,
                    text_emb_dim,
                    decoder,
                    num_decoder_heads,
                    protein_num_heads,
                    perceiver_latent_size,
                    num_perceiver_layers,
                    enable_gradient_checkpointing,
                    dropout,
                )
                if layer_id in self.layers_to_cross_attend
                else CPrtDummyCrossAttention(decoder)
                for layer_id, decoder in enumerate(decoder_layers)
            ]
        )
        self.cprt_llm.transformer.h = cross_attention_block

    def _set_layers_to_cross_attend(self, layers_to_cross_attend: List[int] | None) -> None:
        """Check, correct and set layers_to_cross_attend."""
        if layers_to_cross_attend is None:
            layers_to_cross_attend = list(range(self.n_llm_layers))
        assert (
            isinstance(layers_to_cross_attend, list)
            and all([isinstance(i, int) for i in layers_to_cross_attend])
            and max(layers_to_cross_attend) < self.n_llm_layers
        ), "layers_to_cross_attend must be None or list[int]"
        self.layers_to_cross_attend = [i if i >= 0 else self.n_llm_layers + i for i in layers_to_cross_attend]

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
        """Use protein embeddings as encoder_hidden_states for cross attention."""
        with torch.no_grad():
            protein_embeddings = self.esm(protein_ids)

        return self.cprt_llm(
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

    @torch.no_grad()
    def generate(
        self, protein_ids: Tensor, info_ids: Tensor, generation_length: int = 20, keep_prompt: bool = False
    ) -> List[str]:
        """Generate using input_ids and protein_embeddings as encoder_hidden_states."""
        self.eval()
        protein_embeddings = self.esm(protein_ids)
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
