from typing import Optional, Tuple, List, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import LlamaModel, LlamaConfig
from transformers.modeling_outputs import Seq2SeqModelOutput


class ProtNLA(PreTrainedModel):
    def __init__(self, encoder_config: LlamaConfig, decoder_config: LlamaConfig):    
        if encoder_config.is_decoder:
            raise ValueError("The `LlamaEncoder` should set `config.is_decoder=False`")

        if not decoder_config.is_decoder:
            raise ValueError("The `LlamaDecoder` should set `config.is_decoder=True`")
        
        if encoder_config.hidden_size != decoder_config.hidden_size:
            raise ValueError("Encoder and Decoder MUST have the same `hidden_size`") 

        self.encoder = LlamaModel(config=encoder_config)
        self.decoder = LlamaModel(config=decoder_config)
        
        self.generator = nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, False)

        self.post_init()

    def get_encoder(self) -> LlamaModel:
        return self.encoder
    
    def get_decoder(self) -> LlamaModel:
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
            pass