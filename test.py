import torch as th
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm

from prot_llama.lm_modeling import LlamaForMaskedLM

def main():
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    pretrained_model_path = "data/artifacts/checkpoint-prot_llama:v64"
    tokenizer_path = "artifacts/protein_tokenizer"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    model = LlamaForMaskedLM.from_pretrained(pretrained_model_path).to(device)
    model.eval()
    




if __name__ == "__main__":
    main()
