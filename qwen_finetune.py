import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os


def preprocess_data(examples, tokenizer, max_length=512):
    """
    数据预处理函数：将输入输出拼接成模型的输入格式，并进行分词。

    Args:
        examples (dict): 一个字典，包含 `input` 和 `output` 键。
        tokenizer (AutoTokenizer): Qwen 模型的分词器。
        max_length (int): 最大序列长度。

    Returns:
        dict: 包含 tokenized 输入和标签。
    """
    inputs = [ex["input"] for ex in examples]
    outputs = [ex["output"] for ex in examples]

    tokenized = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        outputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    tokenized["labels"] = labels["input_ids"]

    return tokenized


def main():
    version = '0.5B'
    model_name = f"Qwen/Qwen2.5-{version}"  # Or "Qwen/Qwen-1.5B"
    max_length = 128
    batch_size = 8
    num_epochs = 3
    learning_rate = 5e-5
    output_dir = ".ckpt/qwen_finetuned/case/"
    case = 'en_in'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    dataset = load_dataset("json", data_files={
        "train": f"./mt/json_data/train_{case}.json", 
        "test": f"./mt/json_data/test_{case}.json",
        'valid': f"./mt/json_data/test_{case}.json"
        })

    # Data Pre-process
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_data(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()