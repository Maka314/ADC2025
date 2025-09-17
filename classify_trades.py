import os
import subprocess
import json
import sys
import requests  # 千问API使用requests
import csv

# ===== 千问大模型API配置 =====
QW_API_KEY = "sk-8c6f9c9989ec4ac39f9869668a492f47"  # 在此处填写你的千问API Key
QW_API_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # 千问国际站API地址
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
    """调用千问大模型API，返回分类结果（martin/kelly/other）"""
    with open(feature_path, "r") as f:
        features = f.read()
    prompt = f"你是一名资深量化交易分析师。请根据以下交易特征，判断该账户的主要交易策略属于以下哪一类，并简要说明理由： 马丁格尔策略 凯利公式策略 其他策略 交易特征如下：\n{features} \n 你只能输出: martin, kelly, other三个中的一种"
    headers = {
        "Authorization": f"Bearer {QW_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "qwen-plus",  # 可根据实际模型名调整
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(QW_API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        answer = result["choices"][0]["message"]["content"].strip().lower()
        # 只保留martin/kelly/other
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
    results = []
    for fname in os.listdir(test_data_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(test_data_dir, fname)
        feature_path = os.path.join(output_dir, fname.replace(".csv", "_features.json"))
        print(f"处理: {fname}")
        if extract_features(fpath, feature_path):
            label = call_llm(feature_path)
            print(f"{fname} 分类结果: {label}")
            results.append({"file": fname, "label": label})
        else:
            print(f"跳过: {fname}")

    output_csv = "classification_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n所有分类结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()
