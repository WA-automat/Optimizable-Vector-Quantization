import time
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from ovq.utils.Dataset import DataArguments, ModelArguments, TrainingArguments

if __name__ == '__main__':
    MAX_NEW_TOKENS = 128

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
    print("int8")
    start = time.time()

    text = 'Hamburg is in which country?\n'

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)

    end = time.time()
    print(end - start)
