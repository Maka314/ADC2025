import pandas as pd
import numpy as np
import sys
import json


# 用法: python extract_trade_features.py <csv_file_path>
def extract_features(csv_file):
    df = pd.read_csv(csv_file)
    # 假设列名包含: 时间,方向,手数,盈亏
    # 兼容英文列名
    columns = df.columns.str.lower()

    def find_col(keywords, columns):
        matches = [c for c in columns if any(k in c for k in keywords)]
        if not matches:
            raise ValueError(
                f"未找到包含关键字 {keywords} 的列，实际列名: {list(columns)}"
            )
        return matches[0]

    time_col = find_col(["时间", "time"], columns)
    direction_col = find_col(["方向", "dir", "cmd"], columns)
    volume_col = find_col(["手数", "volume", "lot"], columns)
    profit_col = find_col(["盈亏", "profit"], columns)

    # 交易总数
    total_trades = int(len(df))
    # 确保volume列为float类型
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
    # 平均持仓手数
    avg_volume = float(df[volume_col].mean())
    # 最大单笔手数
    max_volume = float(df[volume_col].max())
    # 手数变化统计特征
    volume_changes_series = df[volume_col].diff().fillna(0)
    volume_changes_mean = float(volume_changes_series.mean())
    volume_changes_std = float(volume_changes_series.std())
    volume_changes_max = float(volume_changes_series.max())
    volume_changes_min = float(volume_changes_series.min())
    # 是否有加倍行为
    try:
        double_volume = any(
            np.isclose(df[volume_col][1:].values, 2 * df[volume_col][:-1].values)
        )
    except Exception:
        double_volume = False
    # 盈亏分布
    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0)
    total_profit = float(df[profit_col].sum())
    max_profit = float(df[profit_col].max())
    max_loss = float(df[profit_col].min())
    win_trades = int((df[profit_col] > 0).sum())
    loss_trades = int((df[profit_col] < 0).sum())
    win_rate = float(win_trades / total_trades) if total_trades > 0 else 0.0
    # 最大连续亏损（回撤）
    profits = df[profit_col].values.astype(float)
    cum_profits = np.cumsum(profits)
    drawdown = np.maximum.accumulate(cum_profits) - cum_profits
    max_drawdown = float(drawdown.max())
    # 持仓周期统计特征（假设时间已排序）
    holding_periods_series = (
        pd.to_datetime(df[time_col]).diff().dt.total_seconds().fillna(0)
    )
    holding_periods_mean = float(holding_periods_series.mean())
    holding_periods_std = float(holding_periods_series.std())
    holding_periods_max = float(holding_periods_series.max())
    holding_periods_min = float(holding_periods_series.min())

    features = {
        "total_trades": total_trades,
        "avg_volume": avg_volume,
        "max_volume": max_volume,
        # volume_changes统计特征
        "volume_changes_mean": volume_changes_mean,
        "volume_changes_std": volume_changes_std,
        "volume_changes_max": volume_changes_max,
        "volume_changes_min": volume_changes_min,
        "double_volume_pattern": bool(double_volume),
        "total_profit": total_profit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        # holding_periods_seconds统计特征
        "holding_periods_mean": holding_periods_mean,
        "holding_periods_std": holding_periods_std,
        "holding_periods_max": holding_periods_max,
        "holding_periods_min": holding_periods_min,
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
