from typing import Optional, Tuple, Union
import torch
from torch import nn

from transformers import PreTrainedModel, LlamaConfig, LlamaModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPast


class LlamaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LlamaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transform = LlamaPredictionHeadTransform(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class LlamaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LlamaLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LlamaForMaskedLM(PreTrainedModel):

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            raise ValueError(
                "If you want to use `LLamaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.llama = LlamaModel(config)
        self.cls = LlamaOnlyMLMHead(config)
        self.post_init()

    @torch.no_grad()
    def embeddings(self, input_ids: Optional[torch.Tensor] = None,
                   attention_mask: Optional[torch.Tensor] = None) -> BaseModelOutputWithPast:

        return self.llama(
            input_ids,
            attention_mask=attention_mask
        )

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

        outputs = self.llama(
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
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
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
