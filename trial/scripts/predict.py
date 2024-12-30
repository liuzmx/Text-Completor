import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 使用微调后的模型进行预测
def predict_missing_text(text, fine_tuned_model, fine_tuned_tokenizer):
    inputs = fine_tuned_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            inputs.input_ids, max_new_tokens=128, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95
        )

    predicted_text = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(predicted_text)
    return predicted_text


# 加载微调后的模型和分词器
model_name = "./finetuned_qwen2.5_0.5B"
fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_name)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name)

# 示例用法
text = "A map showing the [MASK] battlegrounds in this election, image A map showing the seven states considered to be battlegrounds in this election, image That leaves just a handful of states where either candidate could win."


predicted_middle_part = predict_missing_text(text, fine_tuned_model, fine_tuned_tokenizer)
print(f"Predicted middle part: {predicted_middle_part}")
