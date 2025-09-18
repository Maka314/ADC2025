import pandas as pd
import numpy as np
import sys
import json


# 用法: python extract_trade_features.py <csv_file_path> [output_json_path]
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
    direction_col = find_col(["方向", "dir", "cmd"], columns)  # 可能用不上，但保留
    volume_col = find_col(["手数", "volume", "lot"], columns)
    profit_col = find_col(["盈亏", "profit"], columns)

    # ---- 基础清洗 ----
    # 排序（若未排序）
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    # 转数值
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)

    # ---- 基础统计（与你原版一致或等价）----
    total_trades = int(len(df))
    avg_volume = float(df[volume_col].mean()) if total_trades > 0 else 0.0
    max_volume = float(df[volume_col].max()) if total_trades > 0 else 0.0

    volume_changes_series = df[volume_col].diff().fillna(0.0)
    volume_changes_mean = (
        float(volume_changes_series.mean()) if total_trades > 0 else 0.0
    )
    volume_changes_std = (
        float(volume_changes_series.std(ddof=1)) if total_trades > 1 else 0.0
    )
    volume_changes_max = float(volume_changes_series.max()) if total_trades > 0 else 0.0
    volume_changes_min = float(volume_changes_series.min()) if total_trades > 0 else 0.0

    # 是否有加倍行为：后一次手数≈前一次手数*2 的占比是否超过 1/2
    try:
        if total_trades >= 2:
            prev_volumes = df[volume_col].iloc[:-1].values
            next_volumes = df[volume_col].iloc[1:].values
            double_volume_flags = np.isclose(
                next_volumes, 2.0 * prev_volumes, rtol=1e-6, atol=1e-12
            )
            double_volume_count = int(np.sum(double_volume_flags))
            double_volume = (
                double_volume_count > 0
                and double_volume_count >= (total_trades - 1) / 2.0
            )
        else:
            double_volume = False
    except Exception:
        double_volume = False

    total_profit = float(df[profit_col].sum())
    max_profit = float(df[profit_col].max()) if total_trades > 0 else 0.0
    max_loss = float(df[profit_col].min()) if total_trades > 0 else 0.0

    win_mask = df[profit_col] > 0
    loss_mask = df[profit_col] < 0
    win_trades = int(win_mask.sum())
    loss_trades = int(loss_mask.sum())
    win_rate = float(win_trades / total_trades) if total_trades > 0 else 0.0

    profits = df[profit_col].values.astype(float)
    cum_profits = np.cumsum(profits)
    drawdown = np.maximum.accumulate(cum_profits) - cum_profits
    max_drawdown = float(drawdown.max()) if total_trades > 0 else 0.0

    holding_periods_series = df[time_col].diff().dt.total_seconds().fillna(0.0)
    holding_periods_mean = (
        float(holding_periods_series.mean()) if total_trades > 0 else 0.0
    )
    holding_periods_std = (
        float(holding_periods_series.std(ddof=1)) if total_trades > 1 else 0.0
    )
    holding_periods_max = (
        float(holding_periods_series.max()) if total_trades > 0 else 0.0
    )
    holding_periods_min = (
        float(holding_periods_series.min()) if total_trades > 0 else 0.0
    )

    # ---- 识别 Kelly 需要的附加特征 ----
    # 1) 胜负分布与盈亏结构
    wins = df.loc[win_mask, profit_col].values
    losses = df.loc[loss_mask, profit_col].values  # 负值
    avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0  # 负数或 0
    median_win = float(np.median(wins)) if wins.size > 0 else 0.0
    median_loss = float(np.median(losses)) if losses.size > 0 else 0.0

    sum_wins = float(np.sum(wins)) if wins.size > 0 else 0.0
    sum_losses_abs = float(np.sum(np.abs(losses))) if losses.size > 0 else 0.0
    profit_factor = (
        float(sum_wins / sum_losses_abs)
        if sum_losses_abs > 0
        else (np.inf if sum_wins > 0 else 0.0)
    )

    # 平均盈亏比（b = 平均盈利 / 平均亏损绝对值）
    payoff_ratio = (
        float(avg_win / abs(avg_loss))
        if avg_loss < 0
        else (np.inf if avg_win > 0 else 0.0)
    )

    # 期望收益（每笔）
    expectancy_per_trade = float(
        win_rate * avg_win + (1 - win_rate) * (avg_loss if avg_loss != 0 else 0.0)
    )

    # 2) Kelly 估算（理论公式 f* = p - q/b）
    # p = 胜率；q = 1 - p；b = 赔率（avg_win / |avg_loss|）
    p = win_rate
    q = 1.0 - p
    b = payoff_ratio if np.isfinite(payoff_ratio) else np.nan
    if b is None or not np.isfinite(b) or b <= 0:
        kelly_f_star = np.nan
    else:
        kelly_f_star = float(p - q / b)
    # 可选：裁剪到 [0, 1] 便于下游使用
    if np.isnan(kelly_f_star):
        kelly_f_star_clipped = np.nan
    else:
        kelly_f_star_clipped = float(np.clip(kelly_f_star, 0.0, 1.0))

    # 3) 仓位对“优势”(edge) 的响应（Kelly 的关键行为特征）
    # 定义一个简单的“滚动优势”指标：prior_rolling_edge_wN
    # 即：到 t-1 为止最近 N 笔交易中，胜=1负=0 的均值（不把当前这笔算进去以避免前视偏差）
    def rolling_edge(prior_outcomes, window):
        # prior_outcomes: 0/1 序列（到 t-1）
        s = pd.Series(prior_outcomes, dtype=float)
        # 向后移动一位，确保“先有结果，后有仓位”对应
        shifted = s.shift(1)
        return shifted.rolling(window=window, min_periods=max(5, window // 3)).mean()

    outcome01 = (df[profit_col] > 0).astype(float)  # 赢=1, 亏=0
    rolling_edges = {}
    for w in (20, 50, 100):
        rolling_edges[w] = rolling_edge(outcome01, w)

    # 计算相关性：当前仓位 与 之前滚动优势
    def safe_corr(a: pd.Series, b: pd.Series):
        valid = a.notna() & b.notna()
        if valid.sum() < 10:
            return np.nan
        return float(a[valid].corr(b[valid]))

    corr_volume_with_prior_rolling_edge_w20 = safe_corr(
        df[volume_col], rolling_edges[20]
    )
    corr_volume_with_prior_rolling_edge_w50 = safe_corr(
        df[volume_col], rolling_edges[50]
    )
    corr_volume_with_prior_rolling_edge_w100 = safe_corr(
        df[volume_col], rolling_edges[100]
    )

    # 4) 亏/赢后仓位变动（Kelly 倾向在“优势”提升时扩仓）
    vol_change = df[volume_col].diff()  # 本笔 - 上一笔
    prev_is_loss = outcome01.shift(1) == 0.0
    prev_is_win = outcome01.shift(1) == 1.0

    vol_change_after_loss_mean = (
        float(vol_change[prev_is_loss].mean()) if prev_is_loss.any() else np.nan
    )
    vol_change_after_win_mean = (
        float(vol_change[prev_is_win].mean()) if prev_is_win.any() else np.nan
    )

    prob_vol_increase_after_loss = (
        float((vol_change[prev_is_loss] > 0).mean()) if prev_is_loss.any() else np.nan
    )
    prob_vol_increase_after_win = (
        float((vol_change[prev_is_win] > 0).mean()) if prev_is_win.any() else np.nan
    )

    # 5) 仓位与先前权益的相关性（权益上升 -> 更敢投，常见于比例下注/准Kelly）
    prior_equity = pd.Series(cum_profits).shift(1)  # 避免前视
    corr_volume_with_prior_equity = safe_corr(df[volume_col], prior_equity)

    # 6) 参考：仓位自相关（有无“惯性”）
    volume_autocorr = np.nan
    if total_trades >= 3:
        try:
            volume_autocorr = float(pd.Series(df[volume_col]).autocorr(lag=1))
        except Exception:
            volume_autocorr = np.nan

    features = {
        # ------ 基础特征（原有） ------
        "total_trades": total_trades,
        "avg_volume": avg_volume,
        "max_volume": max_volume,
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
        "holding_periods_mean": holding_periods_mean,
        "holding_periods_std": holding_periods_std,
        "holding_periods_max": holding_periods_max,
        "holding_periods_min": holding_periods_min,
        # ------ Kelly 识别：盈亏结构 ------
        "avg_win": avg_win,
        "avg_loss": avg_loss,  # 通常为负
        "median_win": median_win,
        "median_loss": median_loss,  # 通常为负
        "payoff_ratio": payoff_ratio,  # b = avg_win / |avg_loss|
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy_per_trade,
        # ------ Kelly 识别：f* 估算 ------
        "kelly_b": b,  # 赔率
        "kelly_f_star": kelly_f_star,  # 可能为 NaN
        "kelly_f_star_clipped": kelly_f_star_clipped,
        # ------ Kelly 识别：仓位-优势联动 ------
        "vol_change_after_loss_mean": vol_change_after_loss_mean,
        "vol_change_after_win_mean": vol_change_after_win_mean,
        "prob_vol_increase_after_loss": prob_vol_increase_after_loss,
        "prob_vol_increase_after_win": prob_vol_increase_after_win,
        "corr_volume_with_prior_rolling_edge_w20": corr_volume_with_prior_rolling_edge_w20,
        "corr_volume_with_prior_rolling_edge_w50": corr_volume_with_prior_rolling_edge_w50,
        "corr_volume_with_prior_rolling_edge_w100": corr_volume_with_prior_rolling_edge_w100,
        "corr_volume_with_prior_equity": corr_volume_with_prior_equity,
        # ------ 参考 ------
        "volume_autocorr_lag1": volume_autocorr,
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
