import pytest
import transformers
from transformers import Trainer

from ovq.utils.Dataset import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, \
    smart_tokenizer_and_embedding_resize, make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments


@pytest.mark.test_LLama
def test_llama():
    DataArguments.data_path = "./data/alpaca_data.json"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        cache_dir=training_args.cache_dir
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf",
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()