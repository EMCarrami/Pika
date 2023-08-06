from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from cprt.model.helper_modules import CrossAttentionDecoderLayer, TruncatedESM2


class Cprt(nn.Module):
    """Class of cprt model."""

    def __init__(
        self,
        language_model: str = "gpt2",
        protein_model: str = "esm2_t6_8M_UR50D",
        protein_layer_to_keep: int = -1,
    ) -> None:
        """Initialize language and protein encoders."""
        super().__init__()
        self.__get_and_truncate_protein_encoder(protein_model, protein_layer_to_keep)
        self.__get_and_modify_llm(language_model)

    def __get_and_truncate_protein_encoder(
        self, esm_model: str, protein_layer_to_keep: int
    ) -> None:
        esm, _ = torch.hub.load("facebookresearch/esm:main", esm_model)  # type: ignore[no-untyped-call]
        self.esm = TruncatedESM2(esm, protein_layer_to_keep)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{esm_model}")

    def __get_and_modify_llm(self, language_model: str) -> None:
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(language_model)
        self.text_tokenizer.add_special_tokens(
            {"pad_token": "<PAD>", "additional_special_tokens": ["<EOC>", "<PROTEIN>"]}
        )
        self.cprt_llm = GPT2LMHeadModel.from_pretrained(language_model)
        self.__add_new_token_embeddings_to_llm()
        self.__add_cross_attention_to_llm()
        self.__modify_generation_input_to_llm()

    def __add_new_token_embeddings_to_llm(self) -> None:
        """Handle the addition of new tokens to the vocab for the model."""
        self.cprt_llm.resize_token_embeddings(len(self.text_tokenizer))
        # TODO: Fix this to make more sense, currently initializes all to the end of sequence tokens.
        embedding_weight = self.cprt_llm.get_input_embeddings().weight.detach()
        embedding_weight[-3:, :] = embedding_weight[-4, :]
        self.cprt_llm.get_input_embeddings().weight = nn.Parameter(embedding_weight)

    def __add_cross_attention_to_llm(self) -> None:
        """Add Cross-Attention layers to all decoder blocks."""
        protein_emb_size = cast(int, self.esm.embed_tokens.embedding_dim)
        cross_attention_block = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(protein_emb_size, decoder)
                for decoder in self.cprt_llm.transformer.h
            ]
        )
        self.cprt_llm.transformer.h = cross_attention_block

    def __modify_generation_input_to_llm(self) -> None:
        """Update the cprt_llm prepare_inputs_for_generation method."""
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
        text_input: List[str],
        protein_input: List[str],
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[FloatTensor] = None,
        token_type_ids: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        labels: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:

        protein_input_ids = self.protein_tokenizer(
            protein_input, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        protein_embeddings = self.esm(protein_input_ids)
        text_input_ids = self.text_tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        )

        cross_attention_mask = None
        # TODO:
        #   cross_attention_mask to be designed such that only the last question to attend to the protein.
        #   the only benefit I can imagine right now is computational efficiency specially with large contexts
        #   the disadvantage is that previous chat may direct the attention for the question

        return self.cprt_llm(
            input_ids=text_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=protein_embeddings,
            encoder_attention_mask=cross_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
