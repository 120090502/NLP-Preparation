from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载基础模型和tokenizer
base_model_path = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 添加特殊token（确保与训练时一致）
special_tokens_dict = {"additional_special_tokens": ["<think>", "<answer>"]}
tokenizer.add_special_tokens(special_tokens_dict)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

# ========== 原始模型测试 ==========
print("\n" + "="*50 + " 原始模型测试 " + "="*50)
test_cases = [
    "1+2+3+4 = ",                     # 基础算术
    "1+2+3+4 = ? <think>",      # 带特殊token的格式
    "What is 30% of 270?"             # 稍复杂问题
]

for case in test_cases:
    inputs = tokenizer(case, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    print(f"\nInput: {case}\nOutput: {tokenizer.decode(outputs[0], skip_special_tokens=False)}")

# ========== LoRA适配器加载与验证 ==========
print("\n" + "="*50 + " 加载LoRA适配器 " + "="*50)
lora_path = "/root/autodl-tmp/model_outputs/model_20250420_150308"

# 先不合并，验证适配器是否生效
peft_model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
peft_model.resize_token_embeddings(len(tokenizer))

# 关键检查点
print("\n" + "-"*20 + " LoRA配置验证 " + "-"*20)
print(peft_model.peft_config)  # 检查配置

# 对比测试（启用/禁用适配器）
print("\n" + "-"*20 + " 适配器效果对比测试 " + "-"*20)
test_case = "<think>1+2+3+4 = <answer>"  # 使用训练时的格式

with peft_model.disable_adapter():  # 原始模型
    outputs_raw = peft_model.generate(**tokenizer(test_case, return_tensors="pt").to(device), 
                    max_new_tokens=512)
    
outputs_lora = peft_model.generate(**tokenizer(test_case, return_tensors="pt").to(device), 
                 max_new_tokens=512)

print(f"\n原始模型输出:\n{tokenizer.decode(outputs_raw[0], skip_special_tokens=False)}")
print(f"\nLoRA模型输出:\n{tokenizer.decode(outputs_lora[0], skip_special_tokens=False)}")

# ========== 合并LoRA权重 ==========
print("\n" + "="*50 + " 合并LoRA权重 " + "="*50)
merged_model = peft_model.merge_and_unload()
print(f"Merged model has LoRA layers: {any('lora' in n for n in merged_model.state_dict().keys())}")

# # 最终测试
# print("\n" + "-"*20 + " 合并后模型测试 " + "-"*20)
# for case in test_cases:
#     inputs = tokenizer(case, return_tensors="pt").to(device)
#     outputs = merged_model.generate(**inputs, max_new_tokens=512)
#     print(f"\nInput: {case}\nOutput: {tokenizer.decode(outputs[0], skip_special_tokens=False)}")


###################
# 确保相同的输入格式
input_text = "<think>1+2+3+4 = "
inputs = tokenizer(input_text, return_tensors="pt").to(device)

outputs_lora = peft_model.generate(**inputs, max_new_tokens=512)
print("LoRA 未合并输出:", tokenizer.decode(outputs_lora[0]))

# 合并模型的输出
outputs_merged = merged_model.generate(**inputs, max_new_tokens=512)
print("合并后输出:", tokenizer.decode(outputs_merged[0]))
