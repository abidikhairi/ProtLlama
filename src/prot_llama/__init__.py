import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from prot_llama.configuration_prot_llama import ProtLlamaConfig
from prot_llama.lm_modeling import ProtLlama
