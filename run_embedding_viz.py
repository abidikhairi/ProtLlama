import torch as th
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm

from prot_llama.lm_modeling import LlamaForMaskedLM

def main():
    pretrained_model_path = "data/artifacts/checkpoint-prot_llama:v157"
    tokenizer_path = "artifacts/protein_tokenizer"
    protein_sequences_file = "data/uniprot/uniprotkb_sequence_AND_superkingdom.tsv"
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    # data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    sequence_df = pd.read_csv(protein_sequences_file, sep="\t")
    dataset = Dataset.from_pandas(sequence_df.sample(n=2000))
    
    dataset = dataset.map(lambda examples: tokenizer(examples['sequence'], return_tensors="pt", padding=True),
                          num_proc=4,
                          batch_size=512,
                          batched=True,
                          remove_columns=['sequence', '__index_level_0__'])
    
    device = th.device('cuda')
    model = LlamaForMaskedLM.from_pretrained(pretrained_model_path)
    model.eval()
    model.to(device)

    data = {
        "superkingdom": []
    }
    embeddings = []    
    
    for example in tqdm(dataset, desc="Generating embeddings"):

        superkingdom = example.pop('superkingdom')
        example.pop('token_type_ids')
        input_ids = th.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = th.tensor(example['attention_mask']).unsqueeze(0).to(device)
        output = model.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        
        token_indices = th.argwhere(input_ids[0] != tokenizer.pad_token_id).flatten().long()
        token_embed = output.last_hidden_state[:, token_indices, :]
        prot_embed = token_embed.mean(dim=1).flatten().cpu().tolist()

        data['superkingdom'].append(superkingdom)
        embeddings.append(prot_embed)
    
    tsne = TSNE(n_components=2)

    components = tsne.fit_transform(th.tensor(embeddings).numpy())

    embeddings_df = pd.DataFrame(data)
    embeddings_df['x1'] = components[:, 0] 
    embeddings_df['x2'] = components[:, 1]
    
    colors = {
        'Eukaryota': "green",
        'Archaea': "blue",
        'Viruses': "red",
        'Bacteria': "purple"
    }

    plt.figure(figsize=(10, 8))
    for label, color in colors.items():
        sub_df = embeddings_df[embeddings_df['superkingdom'] == label]
        
        plt.scatter(sub_df['x1'], sub_df['x2'], c=color,
                    label=label, cmap="Dark2", alpha=0.5, edgecolors='none')

    plt.axis('off')
    plt.legend(loc='upper right')
    plt.savefig('data/embeddings_vis.png', transparent=True)
    plt.show()



if __name__ == "__main__":
    main()
