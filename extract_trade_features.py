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
    columns = df.columns.str.lower()

    def find_col(keywords, columns):
        matches = [c for c in columns if any(k in c for k in keywords)]
        if not matches:
            raise ValueError(
                f"未找到包含关键字 {keywords} 的列，实际列名: {list(columns)}"
            )
        return matches[0]

    time_col = find_col(
        ["时间", "time", "timestamp", "open_time", "close_time"], columns
    )
    # 可能用不上，但保留
    direction_col = (
        find_col(["方向", "dir", "cmd", "side"], columns)
        if any(k in c for c in columns for k in ["方向", "dir", "cmd", "side"])
        else None
    )
    volume_col = find_col(["手数", "volume", "lot", "size"], columns)
    profit_col = find_col(["盈亏", "profit", "pnl"], columns)

    # ---- 基础清洗 ----
    # 排序（若未排序）
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    # 转数值
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)

    # ---- 基础统计（与原版兼容）----
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

    # 是否有“加倍”行为（简化马丁格尔强信号）
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

    # ---- 盈亏结构（兼容字段）----
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

    payoff_ratio = (
        float(avg_win / abs(avg_loss))
        if avg_loss < 0
        else (np.inf if avg_win > 0 else 0.0)
    )

    expectancy_per_trade = float(
        win_rate * avg_win + (1 - win_rate) * (avg_loss if avg_loss != 0 else 0.0)
    )

    # ---- Kelly 估算（兼容字段）----
    p = win_rate
    q = 1.0 - p
    b = payoff_ratio if np.isfinite(payoff_ratio) else np.nan
    if b is None or not np.isfinite(b) or b <= 0:
        kelly_f_star = np.nan
    else:
        kelly_f_star = float(p - q / b)
    kelly_f_star_clipped = (
        float(np.clip(kelly_f_star, 0.0, 1.0))
        if (kelly_f_star is not None and np.isfinite(kelly_f_star))
        else np.nan
    )

    # ---- 定义“先验优势”(edge)与“先验权益” ----
    # 先验胜负：赢=1, 亏=0（shift 以防前视）
    outcome01 = (df[profit_col] > 0).astype(float)
    outcome01_shift = outcome01.shift(1)

    def rolling_edge_binary(s01: pd.Series, window: int):
        # prior (t-1) 的胜率滚动均值
        return (
            s01.shift(1).rolling(window=window, min_periods=max(5, window // 3)).mean()
        )

    rolling_edges = {}
    for w in (10, 20, 50, 100):
        rolling_edges[w] = rolling_edge_binary(outcome01, w)

    # 先验权益（避免前视）
    prior_equity = pd.Series(cum_profits).shift(1)

    # ---- 规模无关：用“仓位占权益”来衡量下注比例（近似 f）----
    # 处理权益为 0 的情况，采用 max(|权益|, 分位数) 做稳健缩放
    equity_abs = prior_equity.abs()
    denom = equity_abs.clip(
        lower=(
            np.nanpercentile(equity_abs.dropna(), 5)
            if equity_abs.notna().any()
            else 1.0
        )
    )
    vol_frac_of_equity = df[volume_col] / denom.replace(0, np.nan)
    vol_frac_of_equity = vol_frac_of_equity.replace([np.inf, -np.inf], np.nan)

    # ---- 相关性工具 ----
    import warnings

    def safe_corr(a: pd.Series, b: pd.Series, method="pearson"):
        valid = a.notna() & b.notna()
        if valid.sum() < 10:
            return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return float(a[valid].corr(b[valid], method=method))
        except Exception:
            return np.nan

    # 与先验滚动优势/先验权益的相关性（线性 + 秩相关）
    corr_linear = {}
    corr_spearman = {}
    for w in (10, 20, 50, 100):
        corr_linear[f"w{w}"] = safe_corr(
            df[volume_col], rolling_edges[w], method="pearson"
        )
        corr_spearman[f"w{w}"] = safe_corr(
            df[volume_col], rolling_edges[w], method="spearman"
        )

    corr_volume_with_prior_equity = safe_corr(
        df[volume_col], prior_equity, method="pearson"
    )
    corr_volume_with_prior_equity_spear = safe_corr(
        df[volume_col], prior_equity, method="spearman"
    )

    # 同样计算 “仓位占权益” 与边/权益的相关（更接近 Kelly 的下注比例逻辑）
    corr_frac_linear = {}
    corr_frac_spearman = {}
    for w in (10, 20, 50, 100):
        corr_frac_linear[f"w{w}"] = safe_corr(
            vol_frac_of_equity, rolling_edges[w], method="pearson"
        )
        corr_frac_spearman[f"w{w}"] = safe_corr(
            vol_frac_of_equity, rolling_edges[w], method="spearman"
        )

    corr_frac_with_prior_equity = safe_corr(
        vol_frac_of_equity, prior_equity, method="pearson"
    )
    corr_frac_with_prior_equity_spear = safe_corr(
        vol_frac_of_equity, prior_equity, method="spearman"
    )

    # ---- 赢/亏后的仓位变化方向与强度（方向性）----
    vol_change = df[volume_col].diff()
    prev_is_loss = outcome01_shift == 0.0
    prev_is_win = outcome01_shift == 1.0

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

    # 更强的方向性指标：平均规模变化比
    def mean_ratio(a, b, eps=1e-9):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            v = (a / (b.replace(0, np.nan) + eps)).dropna()
        return float(v.mean()) if len(v) >= 5 else np.nan

    avg_vol_after_win = (
        float(df[volume_col][prev_is_win].mean()) if prev_is_win.any() else np.nan
    )
    avg_vol_after_loss = (
        float(df[volume_col][prev_is_loss].mean()) if prev_is_loss.any() else np.nan
    )
    vol_ratio_win_vs_loss = mean_ratio(
        df[volume_col][prev_is_win], df[volume_col][prev_is_loss]
    )

    # ---- 平滑性/稳定性（Kelly 倾向“平滑调仓”）----
    vol_abs_change = vol_change.abs()
    volume_abs_change_mean = float(vol_abs_change.mean())
    volume_abs_change_median = float(vol_abs_change.median())
    volume_change_iqr = (
        float((vol_change.quantile(0.75) - vol_change.quantile(0.25)))
        if total_trades > 0
        else np.nan
    )

    volume_autocorr_lag1 = np.nan
    if total_trades >= 3:
        try:
            volume_autocorr_lag1 = float(pd.Series(df[volume_col]).autocorr(lag=1))
        except Exception:
            volume_autocorr_lag1 = np.nan

    # 滚动波动性（小 → 更平滑）
    vol_roll_std_w10 = (
        float(df[volume_col].rolling(10, min_periods=5).std().mean())
        if total_trades > 0
        else np.nan
    )
    vol_roll_std_w20 = (
        float(df[volume_col].rolling(20, min_periods=7).std().mean())
        if total_trades > 0
        else np.nan
    )

    # ---- 与理论 Kelly 正部分的“对齐度” ----
    # 用先验窗口估算 p_t、b_t → 计算 f*_t = p_t - q_t / b_t；仅保留正部分 f*_pos
    def rolling_kelly_fraction(outcomes01: pd.Series, pnl: pd.Series, w: int):
        # 用平均盈亏估赔率 b_t（赢时利润均值/亏时亏损绝对值均值）
        s = pd.DataFrame({"o": outcomes01.astype(float), "p": pnl.astype(float)}).shift(
            1
        )
        roll = s.rolling(window=w, min_periods=max(10, w // 2))
        p_hat = roll["o"].mean()  # 胜率
        wins = s["p"].where(s["p"] > 0)
        losses = s["p"].where(s["p"] < 0)
        avg_win_roll = wins.rolling(window=w, min_periods=max(10, w // 2)).mean()
        avg_loss_roll = losses.rolling(
            window=w, min_periods=max(10, w // 2)
        ).mean()  # 负
        b_hat = avg_win_roll / (avg_loss_roll.abs().replace(0, np.nan))
        f_star = p_hat - (1 - p_hat) / b_hat
        f_star = f_star.replace([np.inf, -np.inf], np.nan)
        f_pos = f_star.clip(lower=0.0, upper=1.0)
        return f_pos

    fpos_w = {}
    for w in (20, 50, 100):
        fpos_w[w] = rolling_kelly_fraction(outcome01, df[profit_col], w)

    # 相关（下注比例 vs f*_pos）与回归斜率（“比例调仓”的强度）
    def safe_beta(x: pd.Series, y: pd.Series):
        # y ~ beta * x（无截距），估计 beta
        valid = x.notna() & y.notna()
        if valid.sum() < 10:
            return np.nan
        xv = x[valid].values
        yv = y[valid].values
        denom = (xv**2).sum()
        if denom <= 0:
            return np.nan
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float((xv * yv).sum() / denom)

    kelly_alignment = {}
    kelly_beta = {}
    for w in (20, 50, 100):
        kelly_alignment[f"w{w}_pearson"] = safe_corr(
            vol_frac_of_equity, fpos_w[w], method="pearson"
        )
        kelly_alignment[f"w{w}_spearman"] = safe_corr(
            vol_frac_of_equity, fpos_w[w], method="spearman"
        )
        kelly_beta[f"w{w}"] = safe_beta(fpos_w[w], vol_frac_of_equity)

    # ---- 回撤敏感度（回撤期是否缩仓：Kelly 常见行为）----
    in_drawdown = drawdown > 0
    avg_frac_in_dd = (
        float(vol_frac_of_equity[in_drawdown].mean()) if in_drawdown.any() else np.nan
    )
    avg_frac_out_dd = (
        float(vol_frac_of_equity[~in_drawdown].mean())
        if (~in_drawdown).any()
        else np.nan
    )
    frac_ratio_out_vs_in_dd = (
        (avg_frac_out_dd / avg_frac_in_dd)
        if (avg_frac_in_dd not in [0, None, np.nan])
        else np.nan
    )

    # ---- 连胜/连亏后的响应 ----
    # 计算 N 连胜/连亏（简单游程长度），看之后的仓位占权益均值
    def run_lengths(binary_series: pd.Series):
        # 返回每个位置的当前游程长度（包括该位）：赢序列为正，亏序列为负
        vals = binary_series.fillna(0).astype(int).values
        out = np.zeros_like(vals, dtype=int)
        curr = 0
        prev = None
        for i, v in enumerate(vals):
            if prev is None or v != prev or v == 0:
                curr = 1 if v == 1 else (-1 if v == 0 else 0)
            else:
                curr = curr + 1 if v == 1 else curr - 1
            out[i] = curr
            prev = v
        return pd.Series(out, index=binary_series.index)

    runs = run_lengths(outcome01_shift)

    def mean_frac_after_run(k):
        # k>0：连胜 k；k<0：连亏 |k|
        mask = runs == k
        return float(vol_frac_of_equity[mask].mean()) if mask.any() else np.nan

    frac_after_2wins = mean_frac_after_run(2)
    frac_after_3wins = mean_frac_after_run(3)
    frac_after_2loss = mean_frac_after_run(-2)
    frac_after_3loss = mean_frac_after_run(-3)

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
        # 基础特征（原有）
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
        # Kelly 识别：盈亏结构（原有）
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "median_win": median_win,
        "median_loss": median_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy_per_trade,
        # Kelly 识别：f* 估算（原有）
        "kelly_b": b,
        "kelly_f_star": kelly_f_star,
        "kelly_f_star_clipped": kelly_f_star_clipped,
        # Kelly 识别：赢/亏后的仓位变动（原有基础 + 强化）
        "vol_change_after_loss_mean": vol_change_after_loss_mean,
        "vol_change_after_win_mean": vol_change_after_win_mean,
        "prob_vol_increase_after_loss": prob_vol_increase_after_loss,
        "prob_vol_increase_after_win": prob_vol_increase_after_win,
        # 先验权益线性相关（原有）
        "corr_volume_with_prior_equity": corr_volume_with_prior_equity,
        # 参考（原有）
        "volume_autocorr_lag1": volume_autocorr_lag1,
        # 新增：前十条原始交易记录（csv格式字符串）
        "records_head10": records_head10_csv,
    }

    # 新增：多窗口“仓位 vs 滚动优势”的相关性（线性/秩）
    features.update(
        {
            "corr_volume_with_prior_rolling_edge_w10": corr_linear["w10"],
            "corr_volume_with_prior_rolling_edge_w20": corr_linear["w20"],
            "corr_volume_with_prior_rolling_edge_w50": corr_linear["w50"],
            "corr_volume_with_prior_rolling_edge_w100": corr_linear["w100"],
            "corr_spearman_volume_edge_w10": corr_spearman["w10"],
            "corr_spearman_volume_edge_w20": corr_spearman["w20"],
            "corr_spearman_volume_edge_w50": corr_spearman["w50"],
            "corr_spearman_volume_edge_w100": corr_spearman["w100"],
            "corr_spearman_volume_prior_equity": corr_volume_with_prior_equity_spear,
        }
    )

    # 新增：“仓位占权益” 与 边/权益 的相关（Kelly 下注比例更匹配）
    features.update(
        {
            "kelly_corr_frac_edge_w10": corr_frac_linear["w10"],
            "kelly_corr_frac_edge_w20": corr_frac_linear["w20"],
            "kelly_corr_frac_edge_w50": corr_frac_linear["w50"],
            "kelly_corr_frac_edge_w100": corr_frac_linear["w100"],
            "kelly_spear_frac_edge_w10": corr_frac_spearman["w10"],
            "kelly_spear_frac_edge_w20": corr_frac_spearman["w20"],
            "kelly_spear_frac_edge_w50": corr_frac_spearman["w50"],
            "kelly_spear_frac_edge_w100": corr_frac_spearman["w100"],
            "kelly_corr_frac_prior_equity": corr_frac_with_prior_equity,
            "kelly_spear_frac_prior_equity": corr_frac_with_prior_equity_spear,
        }
    )

    # 新增：平滑性/稳定性度量
    features.update(
        {
            "volume_abs_change_mean": volume_abs_change_mean,
            "volume_abs_change_median": volume_abs_change_median,
            "volume_change_iqr": volume_change_iqr,
            "volume_roll_std_w10_mean": vol_roll_std_w10,
            "volume_roll_std_w20_mean": vol_roll_std_w20,
        }
    )

    # 新增：赢/亏后的平均仓位与比率（方向性更强）
    features.update(
        {
            "avg_volume_after_win": avg_vol_after_win,
            "avg_volume_after_loss": avg_vol_after_loss,
            "volume_ratio_win_vs_loss": vol_ratio_win_vs_loss,
        }
    )

    # 新增：与正部分 Kelly 分数的对齐度（相关 & 回归斜率）
    features.update(
        {
            "kelly_alignment_w20_pearson": kelly_alignment["w20_pearson"],
            "kelly_alignment_w20_spearman": kelly_alignment["w20_spearman"],
            "kelly_alignment_w50_pearson": kelly_alignment["w50_pearson"],
            "kelly_alignment_w50_spearman": kelly_alignment["w50_spearman"],
            "kelly_alignment_w100_pearson": kelly_alignment["w100_pearson"],
            "kelly_alignment_w100_spearman": kelly_alignment["w100_spearman"],
            "kelly_beta_w20": kelly_beta["w20"],
            "kelly_beta_w50": kelly_beta["w50"],
            "kelly_beta_w100": kelly_beta["w100"],
        }
    )

    # 新增：回撤敏感度 & 连胜/连亏响应
    features.update(
        {
            "kelly_avg_frac_in_drawdown": avg_frac_in_dd,
            "kelly_avg_frac_out_drawdown": avg_frac_out_dd,
            "kelly_frac_ratio_out_vs_in_dd": frac_ratio_out_vs_in_dd,
            "kelly_frac_after_2wins": frac_after_2wins,
            "kelly_frac_after_3wins": frac_after_3wins,
            "kelly_frac_after_2loss": frac_after_2loss,
            "kelly_frac_after_3loss": frac_after_3loss,
        }
    )

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
