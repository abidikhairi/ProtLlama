"""Models ready for the ProtGym benchmark"""
from typing import Optional
import torch as th
from torch import nn
from prot_llama.modeling_llama import ProtLlamaEncoder

class ProtLlamaForProteinClassfication(nn.Module):
    def __init__(self, encoder_model_path: str, num_classes: int = 1):
        super().__init__()

        self.encoder = ProtLlamaEncoder.from_pretrained(encoder_model_path)
        self.encoder.eval()

        self.cls = nn.Linear(self.encoder.config.hidden_size, num_classes, False)

    def forward(self, input_ids: th.Tensor, attention_mask: Optional[th.Tensor] = None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x =  h.last_hidden_state
        import pdb; pdb.set_trace()
        return self.cls(x)
