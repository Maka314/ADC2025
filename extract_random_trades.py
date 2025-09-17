import os
import random
import re
import shutil

# 策略与目录映射
dir_map = {
    "kelly": "kelly_tradings",
    "martingale": "martingale_tradings",
    "login": "login_tradings",
}

# 目标文件夹
output_dir = "test_data"
os.makedirs(output_dir, exist_ok=True)

# 提取id的正则
id_pattern = re.compile(r"(\d+)")

# 每个策略抽取数量
sample_num = 200

for strategy, folder in dir_map.items():
    folder_path = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder_path):
        print(f"目录不存在: {folder_path}")
        continue
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(files) < sample_num:
        print(f"警告: {folder} 只有 {len(files)} 个csv文件，少于{sample_num}个。")
        chosen = files
    else:
        chosen = random.sample(files, sample_num)
    for fname in chosen:
        match = id_pattern.search(fname)
        id_str = match.group(1) if match else "unknown"
        new_name = f"{strategy}——{id_str}.csv"
        src = os.path.join(folder_path, fname)
        dst = os.path.join(output_dir, new_name)
        shutil.copy2(src, dst)
        print(f"复制: {src} -> {dst}")
print("全部完成！")
