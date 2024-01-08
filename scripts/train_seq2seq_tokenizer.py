"""Train a tokenizer"""
import argparse
import pandas as pd
from tokenizers import Tokenizer, processors, models, trainers, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def text_file_iterator(df: pd.DataFrame, text_col_name: str):
    for line in df[text_col_name]:
        yield line.strip()


def main(save_path: str, text_file: str):
    df = pd.read_csv(text_file, sep='\t')
    

    annotation_start_token = '<annotation>'
    annotation_end_token = '</annotation>'
    pad_token = '<pad>'
    unk_token = '<unk>'

    special_tokens = [annotation_start_token, annotation_end_token, pad_token, unk_token]
    initial_alphabet = ['cellular_component', 'biological_process', 'molecular_function']

    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(initial_alphabet=initial_alphabet, show_progress=True,
                                  special_tokens=special_tokens)

    tokenizer.train_from_iterator(text_file_iterator(
        df, "value"), trainer=trainer, length=len(df))

    tokenizer.enable_padding(direction="right", pad_id=tokenizer.token_to_id("<pad>"), pad_token=pad_token)
    
    annotation_start_token_id = tokenizer.token_to_id(annotation_start_token)
    annotation_end_token_id = tokenizer.token_to_id(annotation_end_token)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<annotation>:0 $A:0 </annotation>",
        special_tokens=[(annotation_start_token, annotation_start_token_id), (annotation_end_token, annotation_end_token_id)],
    )

    trained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token=annotation_start_token, eos_token=annotation_end_token,
        unk_token=unk_token, pad_token=pad_token)

    trained_tokenizer.save_pretrained(save_path)

    loaded_tknzr = PreTrainedTokenizerFast.from_pretrained(save_path)
    print(loaded_tknzr)

    print(loaded_tknzr('biological_process membrane raft assembly'))
    print(tokenizer.encode('biological_process membrane raft assembly').tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Creates a Protein FastTokenizer compatible with Huggingfaces")

    parser.add_argument('--text-file', type=str,
                        required=True, help="Sequences file path (each line is a sequence).")
    parser.add_argument('--save-path', type=str,
                        required=True, help="tokenizer save path")

    args = parser.parse_args()

    main(args.save_path, args.text_file)
