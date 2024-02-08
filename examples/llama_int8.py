import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM

from ovq.utils.Dataset import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, \
    smart_tokenizer_and_embedding_resize, make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments
from ovq.utils.metric import compute_metrics

if __name__ == '__main__':
    DataArguments.train_data_path = "./data/train_data.json"
    DataArguments.eval_data_path = "./data/eval_data.json"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
    # n_gpus = torch.cuda.device_count()
    # max_memory = {i: max_memory for i in range(n_gpus)}

    # 使用封装好的 LLM.int8 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "../model/decapoda-research-llama-7B-hf",
        cache_dir=training_args.cache_dir,
        device_map='auto',
        load_in_8bit=True,
        # max_memory=max_memory
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "../model/decapoda-research-llama-7B-hf",
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
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args,
    #                   compute_metrics=compute_metrics,
    #                   **data_module)
    # trainer.train()
    # trainer.evaluate()
    # trainer.save_state()
    # trainer.save_model()
