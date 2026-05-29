import re

def to_valid_folder_name(name: str) -> str:
    name = name.strip()
    # 去掉所有的 .
    name = name.replace('.', '')
    # 替换非法字符（不是字母、数字、-、_）为 -
    name = re.sub(r'[^A-Za-z0-9\-_]', '-', name)
    # 合并连续的 -
    name = re.sub(r'-{2,}', '-', name)
    return name

model_name = "DeepSeek-R1-Distill-Qwen-7B"
name_identifier = to_valid_folder_name(model_name)
print(name_identifier)  

#DeepSeek-R1-Distill-Qwen-7B
#Qwen2.5-7B-Instruct