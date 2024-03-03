"""Llama for Masked Language Modeling"""
import logging
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from prot_llama.modeling_prot_llama import ProtLlamaEncoder
from prot_llama.configuration_prot_llama import ProtLlamaConfig


logger = logging.getLogger(__name__)


class ProtLlamaPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ProtLlamaMLMHead(nn.Module):
    def __init__(self, config: ProtLlamaConfig):
        super().__init__()
        self.predictions = ProtLlamaPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class ProtLlama(PreTrainedModel):
    config_class = ProtLlamaConfig

    def __init__(self, config: ProtLlamaConfig):
        super().__init__(config)

        if config.is_decoder:
            raise ValueError(
                "If you want to use `PortLlama` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.prot_llama = ProtLlamaEncoder(config)
        self.cls = ProtLlamaMLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.prot_llama(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
