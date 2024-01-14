"""Train a tokenizer"""
import argparse
from tokenizers import Tokenizer, processors, models, trainers, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def text_file_iterator(text_file_path: str, annotation_file_path: str):
    with open(text_file_path, 'r', encoding='utf-8') as stream:
        for line in stream:
            yield line.strip()

    with open(annotation_file_path, 'r', encoding='utf-8') as stream:
        for line in stream:
            yield line.strip()



def main(save_path: str, protein_file: str, annotation_file):
    """
    Creates a Protein FastTokenizer compatible with Hugging Face's transformers library and saves it.

    Args:
        save_path (str): The path where the trained tokenizer will be saved.
        text_file (str): The path to the text file that contains protein sequences.

    This script sets up a Protein FastTokenizer using the `tokenizers` library, adds special tokens,
    and saves the trained tokenizer using the `PreTrainedTokenizerFast` from Hugging Face's transformers library.
    The resulting tokenizer is compatible with the Hugging Face's transformers library.

    Usage:
        python train_tokenizer.py --protein-sequences-file /path/to/file.txt --go-annotation-file /path/to/save/tokenizer

    Args:
        --protein-sequences-file (str): The path where the trained tokenizer will be saved.
        --go-annotation-file (str): The path where the trained tokenizer will be saved.
        --text-file (str): The path to the text file that contains protein sequences.
    """
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    protein_start_token = '<protein>'
    protein_end_token = '</protein>'
    annotation_start_token = "<annotation>"
    annotation_end_token = "</annotation>"
    pad_token = '<pad>'
    unk_token = '<unk>'
    biological_process_token = "biological_process"
    molecular_function_token = "molecular_function"
    cellular_component_token = "cellular_component"
    bos_token = "<bos>"
    eos_token = "<eos>"

    initial_alphabets = [biological_process_token, molecular_function_token, cellular_component_token] + amino_acids

    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))

    special_tokens = [protein_start_token, protein_end_token,pad_token, unk_token, annotation_end_token, annotation_start_token,
                      bos_token, eos_token]
    
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()


    trainer = trainers.BpeTrainer(initial_alphabet=initial_alphabets, show_progress=True,
                                  special_tokens=special_tokens)

    tokenizer.train_from_iterator(text_file_iterator(
        protein_file, annotation_file), trainer=trainer, length=100000)

    tokenizer.enable_padding(direction="right", pad_id=tokenizer.token_to_id("<pad>"), pad_token=pad_token)
    
    protein_start_token_id = tokenizer.token_to_id('<protein>')
    protein_end_token_id = tokenizer.token_to_id('</protein>')
    annotation_start_token_id = tokenizer.token_to_id('<annotation>')
    annotation_end_token_id = tokenizer.token_to_id('</annotation>')
    bos_token_id = tokenizer.token_to_id('<bos>')
    eos_token_id = tokenizer.token_to_id('<eos>')

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> <protein> $A:0 </protein> <annotation>",
        pair="<bos> <protein> $A:0 </protein> <annotation> $B:1 </annotation> <eos>",
        special_tokens=[
            (protein_start_token, protein_start_token_id),
            (protein_end_token, protein_end_token_id),
            (bos_token, bos_token_id),
            (eos_token, eos_token_id),
            (annotation_start_token, annotation_start_token_id),
            (annotation_end_token, annotation_end_token_id)],
    )

    trained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token=protein_start_token, eos_token=protein_end_token,
        unk_token=unk_token, pad_token=pad_token)

    trained_tokenizer.save_pretrained(save_path)

    loaded_tknzr = PreTrainedTokenizerFast.from_pretrained(save_path)
    print(loaded_tknzr)

    print(tokenizer.encode("ABGDDCDCDPCD").tokens)
    print(tokenizer.encode("ABGDDCDCDPCD", "biological_process membrane raft assembly").tokens)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Creates a Protein FastTokenizer compatible with Huggingfaces")

    parser.add_argument('--protein-sequences-file', type=str,
                        required=True, help="Sequences file path (each line is a sequence).")
    parser.add_argument('--go-annotation-file', type=str,
                        required=True, help="Gene Ontology names file path (each line is a GO annotation).")
    
    parser.add_argument('--save-path', type=str,
                        required=True, help="tokenizer save path")

    args = parser.parse_args()

    main(args.save_path, args.protein_sequences_file, args.go_annotation_file)
