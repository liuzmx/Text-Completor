import os

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = "/data/modelscope/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 将数据转换为Dataset对象
data = []
with open("train2_10k.txt", "r") as f:
    for line in f.readlines():
        data.append({"text": line.strip()})
dataset = Dataset.from_list(data)


def preprocess_function(smaple):
    input_sequence, target_sequence = smaple["text"].strip().split("\t")
    model_inputs = tokenizer(input_sequence, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_sequence, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=False)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# 开始微调
trainer.train()

# 保存微调后的模型
model.save_pretrained("finetuned_qwen2.5_0.5B")
tokenizer.save_pretrained("finetuned_qwen2.5_0.5B")
