import torch
from vllm import LLM, SamplingParams
import pandas as pd
import os
import json

# --- 全局配置 ---
TEACHER_MODEL_PATH = "./qwen-32b-model"
CSV_PATH = "./Dataset/train.csv"
SAVE_DIR = "./Dataset/teacher_logits_topk" # 改为目录存储，每条数据一个文件
TOP_K = 100  # 只存前 100 个最可能的词

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 1. 加载教师模型 (注意：必须使用教师自己的分词器推理)
# tensor_parallel_size 根据显卡数量设置
llm = LLM(model=TEACHER_MODEL_PATH, tensor_parallel_size=2, trust_remote_code=True)

# 2. 读取数据
df = pd.read_csv(CSV_PATH)
# 假设你的 CSV 有 'prompt' 和 'answer' 两列
# 我们需要对完整的 'prompt + answer' 获取教师的看法
full_texts = (df['prompt'] + df['answer']).tolist()

# 3. 配置 vLLM 提取 Logits
# prompt_logprobs 表示返回输入文本每个 token 对应的 top-k 概率
sampling_params = SamplingParams(
    max_tokens=1,       # 我们不需要生成新内容，只需要看已有的 logits
    prompt_logprobs=TOP_K, 
)

print(f"开始推理，共 {len(full_texts)} 条数据...")
outputs = llm.generate(full_texts, sampling_params)

# 4. 转换并存储（瘦身处理）
for i, output in enumerate(outputs):
    # output.prompt_logprobs 是一个列表，长度等于输入 sequence 的长度
    # 每个元素是一个字典 {token_id: Logprob_obj}
    
    sparse_logits_data = []
    
    for logprob_dict in output.prompt_logprobs:
        if logprob_dict is None: continue
        
        # 提取 top-k 的 logprob 和对应的 token 字符串
        # 为什么要存字符串？因为学生模型的分词器和教师不同，存 ID 没用
        current_step_data = {
            "p": [], # 概率值
            "t": []  # 对应的词字符串
        }
        
        for tid, logprob_obj in logprob_dict.items():
            current_step_data["p"].append(round(logprob_obj.logprob, 4)) # 存 logprob 节省计算
            current_step_data["t"].append(logprob_obj.decoded_token)    # 存解码后的词
            
        sparse_logits_data.append(current_step_data)
    
    # 保存为 json 或 pt，建议每条数据独立保存，方便 dataset 加载
    save_path = os.path.join(SAVE_DIR, f"{i}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sparse_logits_data, f, ensure_ascii=False)

print(f"处理完成，Logits 已存入 {SAVE_DIR}")