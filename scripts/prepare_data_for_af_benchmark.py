import os
import pandas as pd
from Bio import SeqIO


def main():
    dataset_dir = "data/af_benchmark/protein-high-ident"
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]

    data = {
        "id": [],
        "sequence": []
    }

    for file in files:
        for seq in SeqIO.parse(file, format="fasta"):
            data['id'].append(seq.id)
            data['sequence'].append(seq.seq)
        
    pd.DataFrame(data).to_csv('data/af_benchmark/protein_high_ident.tsv', index=False, sep="\t")

if __name__ == '__main__':
    main()
