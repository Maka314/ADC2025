import pandas as pd
import numpy as np
import sys
import json
import io


# 用法: python extract_trade_features.py <csv_file_path> [output_json_path]
def extract_features(csv_file):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    df = pd.read_csv(csv_file)
    # 假设列名包含: 时间,方向,手数,盈亏（兼容英文）

    # ---- 汇总（保持原有字段 + 新字段）----
    # 处理前10条，所有datetime类型转为字符串，避免json序列化不完整，并以csv格式输出
    head10 = df.head(10).copy()
    for col in head10.columns:
        if np.issubdtype(head10[col].dtype, np.datetime64):
            head10[col] = head10[col].astype(str)

    csv_buffer = io.StringIO()
    head10.to_csv(csv_buffer, index=False)
    records_head10_csv = csv_buffer.getvalue()
    features = {
        "records_head10": records_head10_csv,
    }

    return features


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "用法: python extract_trade_features.py <csv_file_path> [output_json_path]"
        )
        sys.exit(1)
    csv_file = sys.argv[1]
    features = extract_features(csv_file)
    if len(sys.argv) >= 3:
        output_json = sys.argv[2]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
    else:
        print("交易特征:")
        for k, v in features.items():
            print(f"{k}: {v}")
