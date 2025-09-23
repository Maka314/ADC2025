import sys, os

# 自动将项目根目录加入 sys.path，确保可以 import experiment.extract_trade_features
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from extract_trade import extract_features
import os
from collections import defaultdict
import json
from tqdm import tqdm
import openai
import time
import csv


def normalize_label(label):
    label = str(label).strip().lower()
    if label in ["martin", "martingale"]:
        return "martin"
    elif label == "kelly":
        return "kelly"
    elif label == "other":
        return "other"
    elif label == "login":
        return "login"
    else:
        return label


def io_llm(client, features, module_name):
    while True:
        try:
            completion = client.chat.completions.create(
                model=module_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a senior quantitative trading analyst. 
Your task is to classify the trading strategy of the account.

Definitions:
- martingale: position size systematically increases (often doubles) after a loss.
- kelly: position size is dynamically adjusted as a fraction of account equity, 
  based on expected win probability and payoff ratio (p, q, b).
- other: any strategy that does not clearly follow martingale or kelly.

Input data (trading features in JSON):
{features}

Classification rules:
1. Focus only on position sizing and its relationship with prior wins/losses or account edge.
2. If positions frequently double after losses → output "martin".
3. If positions show a consistent correlation with account equity, prior performance, 
   or rolling edge estimates → output "kelly".
4. If neither applies → output "other".

Output format:
Return ONLY one word: "martin", "kelly", or "other".
Do not add explanations, reasoning, or extra text.""",
                    },
                ],
            )
            # 提取 message.content
            content = completion.choices[0].message.content
            return content
        except openai.RateLimitError as e:
            # 获取建议等待时间，否则默认2秒
            import re
            import time

            msg = str(e)
            wait_time = 2
            match = re.search(r"try again in ([0-9.]+)s", msg)
            if match:
                wait_time = float(match.group(1)) + 0.5  # 稍微多等一点
            print(f"RateLimitError: 等待{wait_time}秒后重试...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"LLM调用异常: {e}")
            raise


# 这个实验的主要目的是通过zero shot测试, 确认模型对于马丁格尔凯莉和其他交易策略的识别能力
# 实验会在qwen-max，qwen-turbo，qwq-32b，qwen3-235b-a22b上进行

modules_json_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modules.json"
)
with open(modules_json_path, "r", encoding="utf-8") as f:
    experiment_models_list = json.load(f)  # modules are important

# experiment_models_list = experiment_models_list[-1:]

if __name__ == "__main__":
    test_data_dir = os.path.join(os.path.dirname(__file__), "../test_data")
    files = os.listdir(test_data_dir)

    categories = defaultdict(list)
    for filename in tqdm(files, desc="Extracting features"):
        if os.path.isfile(os.path.join(test_data_dir, filename)):
            # 分类依据为文件名中第一个'-'之前的字符串
            category = filename.split(sep="—")[0]
            file_path = os.path.join(test_data_dir, filename)
            features = extract_features(file_path)
            categories[category].append({"filename": filename, "features": features})

    # print(categories["kelly"][0]["features"])

    for module in experiment_models_list:
        module_name, module_key, module_url = (
            module["model_name"],
            module["apikey"],
            module["baseurl"],
        )
        print(f"开始处理模型{module_name}")

        client = openai.OpenAI(
            api_key=module_key,
            base_url=module_url,
        )

        # 初始化混淆矩阵
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for category in categories:
            print(f"开始处理类别{category}")
            for item in tqdm(categories[category], desc=f"Processing {category}"):
                features = item["features"]
                res = io_llm(client, features, module_name)
                res = normalize_label(res)
                item["llm_result"] = res
                print(f"文件{item['filename']}的分类结果是: {res}")
                # 更新混淆矩阵
                confusion_matrix[category][res] += 1

        # 打印混淆矩阵
        print("混淆矩阵：")
        labels = sorted(
            confusion_matrix.keys() | {k for v in confusion_matrix.values() for k in v}
        )
        print("\t" + "\t".join(labels))
        for true_label in labels:
            row = [
                str(confusion_matrix[true_label][pred_label]) for pred_label in labels
            ]
            print(f"{true_label}\t" + "\t".join(row))

        # 保存混淆矩阵到csv（保存到和 py 文件相同的文件夹）
        ts = int(time.time())
        csv_filename = os.path.join(
            os.path.dirname(__file__), f"res_{module_name}_{ts}.csv"
        )
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label"] + labels)
            for true_label in labels:
                row = [
                    confusion_matrix[true_label][pred_label] for pred_label in labels
                ]
                writer.writerow([true_label] + row)
        print(f"混淆矩阵已保存到: {csv_filename}")
