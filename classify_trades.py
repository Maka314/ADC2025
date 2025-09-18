import os
import subprocess
import json
import sys
import requests  # 千问API使用requests
import csv

# ===== 千问大模型API配置 =====
QW_API_KEY = "sk-8c6f9c9989ec4ac39f9869668a492f47"  # 在此处填写你的千问API Key
QW_API_URL = (
    "https://open.bigmodel.cn/api/paas/v4/chat/completions"  # 千问国际站API地址
)
# ============================

test_data_dir = "test_data"
extract_script = "/home/coder/ADC2025/extract_trade_features.py"
output_dir = "test_data_features"
os.makedirs(output_dir, exist_ok=True)


def extract_features(file_path, output_path):
    """调用特征提取脚本"""
    result = subprocess.run(
        ["python3", extract_script, file_path, output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"特征提取失败: {file_path}\n{result.stderr}")
        return False
    return True


def call_llm(feature_path):
    """调用千问大模型API(通过 openai 库），返回分类结果(martin/kelly/other)"""
    from openai import OpenAI

    with open(feature_path, "r") as f:
        features = f.read()
    prompt = f"""
You are a senior quantitative trading analyst. 
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
3. If positions vary smoothly in proportion to edge / equity fraction → output "kelly".
4. If neither applies → output "other".

Output format:
Return ONLY one word: "martin", "kelly", or "other".
Do not add explanations, reasoning, or extra text.
"""
    try:
        client = OpenAI(
            api_key=QW_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-max", messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content.strip().lower()
        print(f"answer from LLM: {answer}")
        if "martin" in answer:
            return "martin"
        elif "kelly" in answer:
            return "kelly"
        else:
            return "other"
    except Exception as e:
        print(f"千问API调用失败: {e}")
        return "other"


def main():
    output_csv = "classification_results.csv"
    first_write = True
    for fname in os.listdir(test_data_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(test_data_dir, fname)
        feature_path = os.path.join(output_dir, fname.replace(".csv", "_features.json"))
        print(f"处理: {fname}")
        if extract_features(fpath, feature_path):
            label = call_llm(feature_path)
            print(f"{fname} 分类结果: {label}")
            # 实时写入csv
            write_header = False
            if first_write:
                write_header = True
                first_write = False
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "label"])
                if write_header:
                    writer.writeheader()
                writer.writerow({"file": fname, "label": label})
        else:
            print(f"跳过: {fname}")
    print(f"\n所有分类结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()
