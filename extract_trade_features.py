import pandas as pd
import numpy as np
import sys


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
    total_trades = len(df)
    # 平均持仓手数
    avg_volume = df[volume_col].mean()
    # 最大单笔手数
    max_volume = df[volume_col].max()
    # 手数变化序列
    volume_changes = df[volume_col].diff().fillna(0).tolist()
    # 是否有加倍行为
    double_volume = any(
        np.isclose(df[volume_col][1:].values, 2 * df[volume_col][:-1].values)
    )
    # 盈亏分布
    total_profit = df[profit_col].sum()
    max_profit = df[profit_col].max()
    max_loss = df[profit_col].min()
    win_trades = (df[profit_col] > 0).sum()
    loss_trades = (df[profit_col] < 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    # 最大连续亏损（回撤）
    profits = df[profit_col].values
    cum_profits = np.cumsum(profits)
    drawdown = np.maximum.accumulate(cum_profits) - cum_profits
    max_drawdown = drawdown.max()
    # 持仓周期（假设时间已排序）
    holding_periods = (
        pd.to_datetime(df[time_col]).diff().dt.total_seconds().fillna(0).tolist()
    )

    features = {
        "total_trades": total_trades,
        "avg_volume": avg_volume,
        "max_volume": max_volume,
        "volume_changes": volume_changes,
        "double_volume_pattern": double_volume,
        "total_profit": total_profit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "holding_periods_seconds": holding_periods,
    }
    return features


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_trade_features.py <csv_file_path>")
        sys.exit(1)
    csv_file = sys.argv[1]
    features = extract_features(csv_file)
    print("交易特征:")
    for k, v in features.items():
        print(f"{k}: {v}")
