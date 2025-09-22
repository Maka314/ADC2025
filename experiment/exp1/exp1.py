import sys, os

# 自动将项目根目录加入 sys.path，确保可以 import experiment.extract_trade_features
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiment.extract_trade_features import extract_features
import os
from collections import defaultdict
import json
from tqdm import tqdm
import openai


def io_llm(client, messages):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages},
        ],
    )
    # 提取 message.content
    content = completion.choices[0].message.content
    return content


# 这个实验的主要目的是通过zero shot测试, 确认模型对于马丁格尔凯莉和其他交易策略的识别能力
# 实验会在qwen-max，qwen-turbo，qwq-32b，qwen3-235b-a22b上进行

modules_json_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modules.json"
)
with open(modules_json_path, "r", encoding="utf-8") as f:
    experiment_models_list = json.load(f)  # modules are important

if __name__ == "__main__":
    test_data_dir = os.path.join(os.path.dirname(__file__), "../test_data")
    files = os.listdir(test_data_dir)

    # categories = defaultdict(list)
    # for filename in tqdm(files, desc="Extracting features"):
    #     if os.path.isfile(os.path.join(test_data_dir, filename)):
    #         # 分类依据为文件名中第一个'-'之前的字符串
    #         category = filename.split(sep="—")[0]
    #         file_path = os.path.join(test_data_dir, filename)
    #         features = extract_features(file_path)
    #         categories[category].append({"filename": filename, "features": features})

    for module in experiment_models_list:
        module_name, module_key, module_url = (
            module["model_name"],
            module["apikey"],
            module["baseurl"],
        )

        client = openai.OpenAI(
            api_key=module_key,
            base_url=module_url,
        )

        res = io_llm(client, "你好")
        print(f"out put from llm is: {res}")
