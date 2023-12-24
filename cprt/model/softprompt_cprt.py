from typing import List, Optional, Tuple, cast

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.model.cprt_model import BaseCPrtModel
from cprt.model.helper_modules import Perceiver


class SoftPromptCPrt(BaseCPrtModel):
    """CPrt model soft-prompted with protein embeddings at input."""

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
        super(SoftPromptCPrt, self).__init__(language_model, protein_model, protein_layer_to_use)
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        protein_emb_dim = cast(int, self.esm.embed_tokens.embedding_dim)
        protein_num_heads = self.esm.layers[0].self_attn.num_heads
        text_emb_dim = self.cprt_llm.transformer.h[0].attn.c_attn.weight.size(0)
        dropout = self.cprt_llm.transformer.h[0].attn.attn_dropout.p
        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.perceiver = Perceiver(
            protein_emb_dim, perceiver_latent_size, text_emb_dim, protein_num_heads, num_perceiver_layers, dropout
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
        """Add protein latents as word embedding to the beginning of the sentence."""
        assert attention_mask is not None
        assert labels is not None

        info_embeddings = self.cprt_llm.transformer.wte(info_ids)
        with torch.no_grad():
            protein_embeddings = self.esm(protein_ids)
        protein_embeddings = self.protein_layer_norm(protein_embeddings)
        if self.training and self.enable_gradient_checkpointing:
            protein_latents = checkpoint(self.perceiver, protein_embeddings)
        else:
            protein_latents = self.perceiver(protein_embeddings)

        out = self.cprt_llm(
            inputs_embeds=torch.concat([protein_latents, info_embeddings], dim=1),
            past_key_values=past_key_values,
            attention_mask=torch.concat(
                [attention_mask.new_full(size=(protein_latents.shape[:2]), fill_value=1), attention_mask], dim=1
            ),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=torch.concat([labels.new_full(size=(protein_latents.shape[:2]), fill_value=-100), labels], dim=1),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # adjust logits to remove soft-prompts
        out["logits"] = out["logits"][:, protein_latents.size(1) :]
        return out

    @torch.no_grad()
    def generate(
        self, protein_ids: Tensor, info_ids: Tensor, generation_length: int = 20, keep_prompt: bool = False
    ) -> List[str]:
        """Generate using input_embeds and input ids with use_cache==True."""
        self.eval()
        info_embeddings = self.cprt_llm.transformer.wte(info_ids)
        protein_embeddings = self.esm(protein_ids)
        protein_embeddings = self.protein_layer_norm(protein_embeddings)
        protein_latents = self.perceiver(protein_embeddings)
        out = []
        for q_ids, q_embs, protein in zip(info_ids, info_embeddings, protein_latents):
            mask = q_ids == self.text_tokenizer.eos_token_id
            prompt_len = cast(int, torch.where(mask)[0][0] if mask.any() else len(q_ids))
            pos_offset = 0 if keep_prompt else prompt_len
            preds = self.cprt_llm.generate(
                input_ids=torch.concat(
                    [q_ids.new_full(size=(protein.shape[:1]), fill_value=0), q_ids[:prompt_len]]
                ).unsqueeze(0),
                inputs_embeds=torch.concat([protein, q_embs], dim=0).unsqueeze(0),
                use_cache=True,
                max_length=generation_length + prompt_len + len(protein),
            )
            out.append(
                self.text_tokenizer.decode(preds[0, len(protein) + pos_offset :].cpu(), skip_special_tokens=True)
            )
        return out
