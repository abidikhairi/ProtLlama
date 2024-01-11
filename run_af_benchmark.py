import torch as th
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from prot_llama.lm_modeling import LlamaForMaskedLM


def generate_prot_emebdding_vector(sequence, model, tokenizer, device):
        example = tokenizer(sequence)
        example.pop('token_type_ids')

        input_ids = th.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = th.tensor(example['attention_mask']).unsqueeze(0).to(device)
        
        output = model.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        
        token_indices = th.argwhere(input_ids[0] != tokenizer.pad_token_id).flatten().long()
        token_embed = output.last_hidden_state[:, token_indices, :]
        prot_embed = token_embed.mean(dim=1).flatten().cpu().tolist()
        
        return prot_embed


def main():
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    pretrained_model_path = "data/artifacts/checkpoint-prot_llama:v157"
    tokenizer_path = "artifacts/protein_tokenizer"
    dataset_path = "data/af_benchmark/protein_high_ident.tsv"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    dataset = pd.read_csv(dataset_path, sep='\t')

    model = LlamaForMaskedLM.from_pretrained(pretrained_model_path).to(device)
    model.eval()

    submission_data = {
        "s1": [],
        "s2": [],
    }
    pdist = th.nn.PairwiseDistance(p=2)
    embeddings_list_reference = []
    embeddings_list_1 = []
    embeddings_list_2 = []

    for _, row1 in tqdm(dataset.iterrows(), total=len(dataset), leave=False):
        seq1 = row1['sequence']
        emb1 = generate_prot_emebdding_vector(seq1, model, tokenizer, device)
        embeddings_list_reference.append(emb1)
    
    for x in tqdm(embeddings_list_reference, leave=False):
        embeddings_list_1.append(x)
        for y in tqdm(embeddings_list_reference, leave=False):
            embeddings_list_2.append(y)
        

    for _, row1 in tqdm(dataset.iterrows(), total=len(dataset), leave=False):
        id1 = row1['id']
        for _, row2 in tqdm(dataset.iterrows(), total=len(dataset), leave=False):
            id2 = row2['id']
            submission_data['s1'].append(id1)
            submission_data['s2'].append(id2)


    submission_df = pd.DataFrame(submission_data)
    embs1 = th.tensor(embeddings_list_1).float()
    embs2 = th.tensor(embeddings_list_2).float()

    distances = pdist(embs1, embs2).numpy()
    import pdb; pdb.set_trace()

    #submission_df['distances'] = distances

    submission_df.to_csv("data/af_benchmark/submission.tsv", sep="\t", header=False, index=False)    

if __name__ == "__main__":
    main()
