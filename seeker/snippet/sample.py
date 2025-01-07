#date: 2025-01-07T16:48:34Z
#url: https://api.github.com/gists/cf3868d2cd57fe99d971e8ffefa65f69
#owner: https://api.github.com/users/neavo

import os
import random
import shutil

import torch
from rich import print
from datasets import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers.utils import is_torch_bf16_gpu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# 模型
MODEL_NAME = "modern_bert"
MODEL_PATH = f"modern_bert"
OUTPUT_PATH = f"output/"
ATTN_IMPLEMENTATION = "sdpa" # sdpa, flex_attention, flash_attention_2, eager

# 训练
SEED = 42
WEIGHT_DECAY = 1 * 1e-5
LEARNING_RATE = 5 * 1e-5
EPOCHS = 1
EVAL_SIZE = 16
BATCH_SIZE = 32
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_SIZE = 256

# 输出
LOG_STEPS = 1
INTERVAL_STEPS = 10
SAVE_TOTAL_LIMIT = 1

# 数据
LENGTH_THRESHOLD = 256

# 加载模型
def load_model() -> PreTrainedModel:
    return AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code = True,
        ignore_mismatched_sizes = True,
        torch_dtype = torch.bfloat16 if is_torch_bf16_gpu_available() == True else torch.float16,
        attn_implementation = ATTN_IMPLEMENTATION
    ).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器
 "**********"d "**********"e "**********"f "**********"  "**********"l "**********"o "**********"a "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"( "**********") "**********"  "**********"- "**********"> "**********"  "**********"P "**********"r "**********"e "**********"T "**********"r "**********"a "**********"i "**********"n "**********"e "**********"d "**********"T "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"F "**********"a "**********"s "**********"t "**********": "**********"
    # https: "**********"
    # 常规 token 是从使用 UTF-8 编码的文本字节序列中学习的 BPE token。
    # 虽然这允许对所有文本进行 token 化，并且不存在未知 token，但在对不常见的文本进行 token 化时，它可能会退回到使用单个字节。
    # 你可能会遇到 UTF-8 解码错误，由于默认错误为 replace ，因此会导致生成不完整的替换字符（�）。
    # 你可以通过将 errors = "ignore" 传递给 decode 函数一次或永远传递给 from_pretrained 函数来更改此行为。
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        errors = "ignore",
        do_lower_case = False,
        local_files_only = True,
    )

# 加载数据集
def load_dataset(tokenizer: "**********":
    print("")
    print("正在加载数据集 ...")
    print("")

    # 加载或者生成数据集
    cache_path = "dataset/pt/cache"
    os.makedirs(cache_path, exist_ok = True)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"o "**********"s "**********". "**********"p "**********"a "**********"t "**********"h "**********". "**********"i "**********"s "**********"d "**********"i "**********"r "**********"( "**********"f "**********"" "**********"{ "**********"c "**********"a "**********"c "**********"h "**********"e "**********"_ "**********"p "**********"a "**********"t "**********"h "**********"} "**********"/ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"d "**********"/ "**********"{ "**********"M "**********"O "**********"D "**********"E "**********"L "**********"_ "**********"N "**********"A "**********"M "**********"E "**********"} "**********"" "**********") "**********": "**********"
        dataset_tokenized = "**********"
    else:
        dataset_tokenized = "**********"
            "sample.txt",
            cache_dir = cache_path
        ).map(
            lambda samples: "**********"
            num_proc = os.cpu_count(),
            batched = True,
            remove_columns = ["text"],
            cache_file_name = f"{cache_path}/map/{MODEL_NAME}.cache",
            load_from_cache_file = True,
        )
        dataset_tokenized.save_to_disk(
            dataset_path = "**********"
            num_proc = os.cpu_count(),
            max_shard_size = "4GB",
        )
        shutil.rmtree(f"{cache_path}/map", ignore_errors = True)
        shutil.rmtree(f"{cache_path}/text", ignore_errors = True)

    # 统计数据
    max_length = "**********"
    total_length = "**********"

    # 拆分数据集
    dataset_dict = "**********"
        seed = SEED,
        shuffle = True,
        test_size = 2048,
        test_indices_cache_file_name = f"dataset/pt/cache/{MODEL_NAME}_indices_eval.cache",
        train_indices_cache_file_name = f"dataset/pt/cache/{MODEL_NAME}_indices_train.cache",

    )
    eval_dataset, train_dataset = dataset_dict.get("test"), dataset_dict.get("train")

    print("")
    print("数据加载已完成 ... 样本如下：")
    print("")
    print_dataset_sample(tokenizer, dataset_tokenized)
    print("")
    print(""
        + f"共加载 {len(dataset_tokenized)} 条数据，其中有效 Token {(total_length / 1000 / 1000): "**********"
        + f"最长条目 {(max_length): "**********":.2f} Token ..."
    )

    return eval_dataset, train_dataset

# 映射函数
def load_dataset_map_function(samples: "**********": PreTrainedTokenizerFast) -> BatchEncoding:
    encodings = "**********"
        samples["text"],
        padding = "max_length",
        truncation = True,
        max_length = LENGTH_THRESHOLD, # 最大长度是包含特殊 ID 在内的，所以不需要增减
        return_attention_mask = True,
        return_special_tokens_mask = "**********"
    )

    # 计算有效的 Token 数量
    encodings["attention_length"] = [sum(item) for item in encodings.get("attention_mask")]

    return encodings

# 打印数据集样本
def print_dataset_sample(tokenizer: "**********": Dataset) -> None:
    if len(dateset) == 0:
        return

    input_ids = dateset[0].get("input_ids")
    input_tokens = "**********"
    attention_mask = dateset[0].get("attention_mask")
    special_tokens_mask = "**********"

    print(f"{"tokens": "**********":<4}\t\t{"attention":<8}\t\t{"special_mask":<6}")
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"x "**********", "**********"  "**********"z "**********", "**********"  "**********"a "**********", "**********"  "**********"b "**********"  "**********"i "**********"n "**********"  "**********"z "**********"i "**********"p "**********"( "**********"i "**********"n "**********"p "**********"u "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********", "**********"  "**********"i "**********"n "**********"p "**********"u "**********"t "**********"_ "**********"i "**********"d "**********"s "**********", "**********"  "**********"a "**********"t "**********"t "**********"e "**********"n "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"m "**********"a "**********"s "**********"k "**********", "**********"  "**********"s "**********"p "**********"e "**********"c "**********"i "**********"a "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"m "**********"a "**********"s "**********"k "**********") "**********": "**********"
        print(f"{x:<8}\t\t{z:<4}\t\t{a:<8}\t\t{b:<6}")

# 打印模型的参数量
def print_model_parameters(model: PreTrainedModel) -> None:
    total = 0
    layer = 0
    embedding = 0

    for name, param in model.named_parameters():
        total = total + param.numel()
        if "embeddings" not in name:
            layer = layer + param.numel()
        else:
            embedding = embedding + param.numel()

    print("")
    print(f"{MODEL_NAME} : layer - {layer / 1e6:.2f} M / embedding - {embedding / 1e6:.2f} M / total - {total / 1e6:.2f} M")
    print("")

# 开始训练
def start_training(model: "**********": PreTrainedTokenizerFast, eval_dataset: Dataset, train_dataset: Dataset) -> None:
    training_args = TrainingArguments(
        # 输出
        report_to = None,
        output_dir = OUTPUT_PATH,
        logging_dir = "logs",
        logging_steps = LOG_STEPS,
        eval_steps = INTERVAL_STEPS,
        save_steps = INTERVAL_STEPS,
        eval_strategy = "steps",
        save_strategy = "steps",

        # 训练
        bf16 = True,
        bf16_full_eval = True,
        optim = "paged_adamw_8bit",
        warmup_ratio = 0.1,
        weight_decay = WEIGHT_DECAY,
        learning_rate = LEARNING_RATE,
        num_train_epochs = EPOCHS,
        lr_scheduler_type = "warmup_stable_decay",
        lr_scheduler_kwargs = {
            "num_decay_steps": int(len(train_dataset) * 0.1 / max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE)) + 1,
            "num_stable_steps": int(len(train_dataset) * 0.8 / max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE)) + 1,
        },
        per_device_eval_batch_size = EVAL_SIZE,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        gradient_accumulation_steps = int(max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE) / BATCH_SIZE),
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = "**********"
            mlm = True,
            mlm_probability = 0.30,
            pad_to_multiple_of = 8,
        ),
        eval_dataset = eval_dataset,
        train_dataset = train_dataset,
        processing_class = "**********"
    )
    trainer.train()

# 主函数
def main() -> None:
    # 固定随机种子
    random.seed(SEED)

    # 加载分词器
    tokenizer = "**********"

    # 加载数据集
    eval_dataset, train_dataset = "**********"

    # 加载模型
    model = load_model()

    # 调整 token_embeddings 的大小
    model.resize_token_embeddings(len(tokenizer))

    # 打印模型的参数量
    print_model_parameters(model)

    # 开始训练
    start_training(model, tokenizer, eval_dataset, train_dataset)

# 主函数
if __name__ == "__main__":
    main()