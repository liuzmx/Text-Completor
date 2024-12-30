import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
)

# 假设的数据集
documents = [
    "This is a document. It contains some information.",
    "Here is another one. This document talks about something else.",
    "And yet another example. The content here is different.",
]

completions = [
    "This is a document. It contains some information that is useful for understanding the topic.",
    "Here is another one. This document talks about something else, providing additional insights.",
    "And yet another example. The content here is different and offers unique perspectives.",
]

# 将数据写入文件
with open("train.txt", "w") as f:
    for doc, comp in zip(documents, completions):
        f.write(f"{doc} {comp}\n")

# 加载预训练的GPT-2模型和分词器
model_name = "/data/modelscope/GPT2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 添加特殊标记
special_tokens_dict = {"pad_token": "<PAD>"}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


# 准备数据集
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)


train_dataset = load_dataset("train.txt", tokenizer)

# 定义Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 微调模型
trainer.train()

# 保存微调后的模型
model.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")


# 使用微调后的模型进行文本补全
def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# 测试模型
test_document = "This is"
predicted_completion = generate_text(test_document, model, tokenizer)
print(predicted_completion.strip())
