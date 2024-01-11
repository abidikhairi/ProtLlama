import os
import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers import LlamaConfig, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import Dataset

from prot_llama.lm_modeling import LlamaForMaskedLM

os.environ["WANDB_PROJECT"] = "hunayn"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def main():
    ## Load Protein tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained('artifacts/protein_tokenizer')

    ## Start Prepare data
    train_df = pd.read_csv("data/uniprot/encoder_training_data.tsv", names=['Sequence'])
    valid_df = pd.read_csv("data/uniprot/encoder_training_data.tsv", names=['Sequence'])
    
    train_data = Dataset.from_pandas(train_df)
    valid_data = Dataset.from_pandas(valid_df)

    train_data = train_data.map(lambda examples: tokenizer(examples['Sequence']), batch_size=128, batched=True, num_proc=5, remove_columns=['Sequence'])
    valid_data = valid_data.map(lambda examples: tokenizer(examples['Sequence']), batch_size=128, batched=True, num_proc=5, remove_columns=['Sequence'])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    ## End Prepare data 

    ## Model Hyperparameters
    d_model = 16
    num_attention_heads = 4
    num_hidden_layers = 4
    num_key_value_heads = 4
    intermediate_size = 32
    attention_dropout = 0.5

    vocab_size = tokenizer.vocab_size
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    config = LlamaConfig(is_decoder=False, hidden_size=d_model, num_attention_heads=num_attention_heads,
                         num_hidden_layers=num_hidden_layers, num_key_value_heads=num_key_value_heads,
                         max_position_embeddings=35000, intermediate_size=intermediate_size,
                         attention_dropout=attention_dropout,
                         vocab_size=vocab_size, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         use_cache=False, pad_token_id=pad_token_id)

    model = LlamaForMaskedLM(config)

    training_args = TrainingArguments(
        output_dir="outputs/prot_llama",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        do_train=True,
        report_to="wandb",
        run_name="prot_llama"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=True)

    results = trainer.evaluate()
    print(results)


if __name__ == '__main__':
    main()
