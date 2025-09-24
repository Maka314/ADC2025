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
    # 字段兼容性处理
    col_map = {
        'volume': ['手数', 'volume', 'Lots', 'lot', 'size'],
        'direction': ['方向', 'direction', 'Side'],
        'pnl': ['盈亏', 'pnl', 'Profit', 'profit'],
        'equity': ['equity', 'Equity', '权益'],
        'edge': ['edge', 'Edge', 'alpha'],
        'kelly': ['kelly', 'Kelly', 'fstar', 'f*'],
        'datetime': ['时间', 'datetime', 'Date', 'date', 'time', 'Time']
    }
    def find_col(possibles):
        for c in possibles:
            if c in df.columns:
                return c
        return None
    volume_col = find_col(col_map['volume'])
    direction_col = find_col(col_map['direction'])
    pnl_col = find_col(col_map['pnl'])
    equity_col = find_col(col_map['equity'])
    edge_col = find_col(col_map['edge'])
    kelly_col = find_col(col_map['kelly'])
    datetime_col = find_col(col_map['datetime'])

    # ---- 汇总（保持原有字段 + 新字段）----
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

    # ========== 1. Sizing Dynamics 特征 ==========
    sizing_features = {}
    if volume_col:
        v = df[volume_col].astype(float)
        # ∆volume 统计
        dv = v.diff().dropna()
        if len(dv) > 0:
            sizing_features['delta_volume_mean'] = dv.mean()
            sizing_features['delta_volume_std'] = dv.std()
            sizing_features['delta_volume_iqr'] = np.subtract(*np.percentile(dv, [75, 25]))
        else:
            sizing_features['delta_volume_mean'] = np.nan
            sizing_features['delta_volume_std'] = np.nan
            sizing_features['delta_volume_iqr'] = np.nan
        # lag-1 自相关
        if len(v) > 1:
            sizing_features['lag1_autocorr'] = v.autocorr(lag=1)
        else:
            sizing_features['lag1_autocorr'] = np.nan
        # 平滑性（IQR, rolling std）
        window = min(10, len(v))
        if window >= 2:
            rolling_std = v.rolling(window=window).std().dropna()
            sizing_features['rolling_std_mean'] = rolling_std.mean()
            sizing_features['rolling_std_iqr'] = np.subtract(*np.percentile(rolling_std, [75, 25]))
        else:
            sizing_features['rolling_std_mean'] = np.nan
            sizing_features['rolling_std_iqr'] = np.nan
    features['sizing_dynamics'] = sizing_features

    # ========== 2. Edge Coupling 特征 ==========
    edge_coupling = {}
    if volume_col:
        v = df[volume_col].astype(float)
        # size_as_fraction_of_equity
        if equity_col and equity_col in df.columns:
            eq = df[equity_col].astype(float)
            size_frac = v / (eq.replace(0, np.nan))
        else:
            size_frac = None
        # rolling edge
        if edge_col and edge_col in df.columns:
            edge = df[edge_col].astype(float)
            # 相关性
            edge_coupling['corr_size_edge'] = v.corr(edge)
            if size_frac is not None:
                edge_coupling['corr_sizeFrac_edge'] = size_frac.corr(edge)
            else:
                edge_coupling['corr_sizeFrac_edge'] = np.nan
        else:
            edge_coupling['corr_size_edge'] = np.nan
            edge_coupling['corr_sizeFrac_edge'] = np.nan
        # 前一时刻权益
        if equity_col and equity_col in df.columns:
            eq = df[equity_col].astype(float)
            eq_shift = eq.shift(1)
            edge_coupling['corr_size_prevEquity'] = v.corr(eq_shift)
            if size_frac is not None:
                edge_coupling['corr_sizeFrac_prevEquity'] = size_frac.corr(eq_shift)
            else:
                edge_coupling['corr_sizeFrac_prevEquity'] = np.nan
        else:
            edge_coupling['corr_size_prevEquity'] = np.nan
            edge_coupling['corr_sizeFrac_prevEquity'] = np.nan
    features['edge_coupling'] = edge_coupling

    # ========== 3. Response to Outcomes 特征 ==========
    response_features = {}
    if volume_col and pnl_col:
        v = df[volume_col].astype(float)
        pnl = df[pnl_col].astype(float)
        v_shift = v.shift(1)
        # 定义win/loss
        win = pnl > 0
        loss = pnl < 0
        # Pr(size ↑ | win)
        win_idx = win[win].index
        loss_idx = loss[loss].index
        pr_up_win = np.nan
        pr_up_loss = np.nan
        if len(win_idx) > 0:
            up_after_win = (v.loc[win_idx] > v_shift.loc[win_idx]).sum()
            pr_up_win = up_after_win / len(win_idx)
        if len(loss_idx) > 0:
            up_after_loss = (v.loc[loss_idx] > v_shift.loc[loss_idx]).sum()
            pr_up_loss = up_after_loss / len(loss_idx)
        response_features['pr_size_up_given_win'] = pr_up_win
        response_features['pr_size_up_given_loss'] = pr_up_loss
        # 胜/负后均值变化
        mean_change_after_win = (v.loc[win_idx] - v_shift.loc[win_idx]).mean() if len(win_idx) > 0 else np.nan
        mean_change_after_loss = (v.loc[loss_idx] - v_shift.loc[loss_idx]).mean() if len(loss_idx) > 0 else np.nan
        response_features['mean_size_change_after_win'] = mean_change_after_win
        response_features['mean_size_change_after_loss'] = mean_change_after_loss
    features['response_to_outcomes'] = response_features

    # # ========== 4. Kelly Alignment 特征 ==========
    # kelly_features = {}
    # if volume_col and kelly_col:
    #     v = df[volume_col].astype(float)
    #     kelly = df[kelly_col].astype(float)
    #     # 只考虑正的kelly
    #     mask_pos_kelly = kelly > 0
    #     v_pos = v[mask_pos_kelly]
    #     kelly_pos = kelly[mask_pos_kelly]
    #     # 相关性
    #     if len(v_pos) > 1 and len(kelly_pos) > 1:
    #         kelly_features['pearson_corr'] = v_pos.corr(kelly_pos, method='pearson')
    #         kelly_features['spearman_corr'] = v_pos.corr(kelly_pos, method='spearman')
    #         kelly_features['kendall_corr'] = v_pos.corr(kelly_pos, method='kendall')
    #         # 无截距β和R2
    #         beta = np.sum(v_pos * kelly_pos) / np.sum(kelly_pos ** 2) if np.sum(kelly_pos ** 2) > 0 else np.nan
    #         y_pred = beta * kelly_pos
    #         ss_res = np.sum((v_pos - y_pred) ** 2)
    #         ss_tot = np.sum((v_pos - v_pos.mean()) ** 2)
    #         r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    #         kelly_features['no_intercept_beta'] = beta
    #         kelly_features['no_intercept_r2'] = r2
    #         # 分位提升（Q4-Q1）
    #         q = kelly_pos.quantile([0.25, 0.75])
    #         q1, q4 = q.iloc[0], q.iloc[1]
    #         uplift = v_pos[kelly_pos >= q4].mean() - v_pos[kelly_pos <= q1].mean()
    #         kelly_features['quantile_uplift_q4_q1'] = uplift
    #     else:
    #         kelly_features['pearson_corr'] = np.nan
    #         kelly_features['spearman_corr'] = np.nan
    #         kelly_features['kendall_corr'] = np.nan
    #         kelly_features['no_intercept_beta'] = np.nan
    #         kelly_features['no_intercept_r2'] = np.nan
    #         kelly_features['quantile_uplift_q4_q1'] = np.nan
    # features['kelly_alignment'] = kelly_features

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
