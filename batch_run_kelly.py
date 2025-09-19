import subprocess
import multiprocessing
import time

# 配置
TOTAL_RUNS = 400
MAX_WORKERS = 12  # 并发进程数，可根据CPU调整
PYTHON_EXEC = "/home/coder/ADC2025/.venv/bin/python"
SCRIPT = "kelly_backtest.py"


def run_once(idx):
    try:
        result = subprocess.run([PYTHON_EXEC, SCRIPT], capture_output=True, text=True)
        print(f"[Run {idx}] Done. Output: {result.stdout.strip()}")
    except Exception as e:
        print(f"[Run {idx}] Error: {e}")

if __name__ == "__main__":
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        pool.map(run_once, range(1, TOTAL_RUNS + 1))
    print("全部凯利策略回测任务已完成！")
