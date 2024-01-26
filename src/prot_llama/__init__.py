import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from prot_llama.modeling_llama import ProtLlamaEncoder
from prot_llama.lm_modeling import ProtLlamaForMaskedLM
