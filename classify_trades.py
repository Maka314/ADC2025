import os
from pathlib import Path

# 自动加载.env.aws环境变量
env_path = Path(__file__).parent / ".env.aws"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
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
    import openai

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
3. If positions show a consistent correlation with account equity, prior performance, 
   or rolling edge estimates → output "kelly".
4. If neither applies → output "other".

Output format:
Return ONLY one word: "martin", "kelly", or "other".
Do not add explanations, reasoning, or extra text.
"""
    import boto3
    # 选择Bedrock模型
    bedrock_model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    import time
    import botocore
    max_retries = 6
    base_delay = 2
    for attempt in range(max_retries):
        try:
            bedrock = boto3.client("bedrock-runtime")
            body = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "anthropic_version": "bedrock-2023-05-31"
            }
            response = bedrock.invoke_model(
                modelId=bedrock_model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )
            result = json.loads(response["body"].read())
            answer = result["content"][0]["text"].strip().lower() if "content" in result and result["content"] else ""
            print(f"answer from LLM: {answer}")
            if "martin" in answer:
                return "martin"
            elif "kelly" in answer:
                return "kelly"
            else:
                return "other"
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException' and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"ThrottlingException: 第{attempt+1}次重试，等待{delay}秒...")
                time.sleep(delay)
                continue
            else:
                print(f"Bedrock API调用失败: {e}")
                return "other"
        except Exception as e:
            print(f"Bedrock API调用失败: {e}")
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
