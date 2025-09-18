import os
import pandas as pd

folder = 'test_data'
stats = {
    'kelly': {'count': 0, 'rows': 0, 'open_times': []},
    'login': {'count': 0, 'rows': 0, 'open_times': []},
    'martingale': {'count': 0, 'rows': 0, 'open_times': []},
}

for fname in os.listdir(folder):
    for key in stats:
        if fname.startswith(key):
            stats[key]['count'] += 1
            fpath = os.path.join(folder, fname)
            try:
                df = pd.read_csv(fpath)
                stats[key]['rows'] += len(df)
                # 兼容不同列名
                open_time_col = None
                for c in df.columns:
                    if str(c).lower() in ['open_time', 'opentime', '开仓时间', '开仓', 'time', '时间']:
                        open_time_col = c
                        break
                if open_time_col:
                    # 判断是否为纯数字（时间戳）
                    sample = df[open_time_col].dropna().astype(str).iloc[:5]
                    is_numeric = sample.str.match(r'^\d+$').all()
                    if is_numeric:
                        times = pd.to_datetime(df[open_time_col], errors='coerce', unit='s')
                    else:
                        times = pd.to_datetime(df[open_time_col], errors='coerce')
                    stats[key]['open_times'].extend(times.dropna().tolist())
            except Exception as e:
                print(f"读取 {fpath} 失败: {e}")
            break

for key, v in stats.items():
    print(f"{key} 文件数量: {v['count']}")
    print(f"{key} 总交易数: {v['rows']}")
    if v['open_times']:
        min_time = min(v['open_times'])
        max_time = max(v['open_times'])
        print(f"{key} open_time范围: {min_time} ~ {max_time}")
    else:
        print(f"{key} open_time范围: 无数据")
