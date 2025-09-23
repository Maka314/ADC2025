import sys, os

# 自动将项目根目录加入 sys.path，确保可以 import experiment.extract_trade_features
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiment.extract_trade import extract_features
import os
from collections import defaultdict
import json
from tqdm import tqdm
import openai
import time
import csv


def normalize_label(label):
    label = str(label).strip().lower()
    if label in ["martin", "martingale"]:
        return "martin"
    elif label == "kelly":
        return "kelly"
    elif label == "other":
        return "other"
    elif label == "login":
        return "login"
    else:
        return label


def io_llm(client, features, module_name):
    while True:
        try:
            completion = client.chat.completions.create(
                model=module_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a senior quantitative trading analyst. 
Your task is to classify the trading strategy of the account.

Definitions:
- martingale: position size systematically increases (often doubles) after a loss.
- kelly: position size is dynamically adjusted as a fraction of account equity, 
  based on expected win probability and payoff ratio (p, q, b).
- other: any strategy that does not clearly follow martingale or kelly.

Classification rules:
1. Focus only on position sizing and its relationship with prior wins/losses or account edge.
2. If positions frequently double after losses → output "martin".
3. If positions show a consistent correlation with account equity, prior performance, 
   or rolling edge estimates → output "kelly".
4. If neither applies → output "other".

Output format:
Return ONLY one word: "martin", "kelly", or "other".
Do not add explanations, reasoning, or extra text.

[Example #1]
Input data (trading features in JSON):
  "records_head10": "broker_id,ticket,login,symbol,cmd,volume,open_time,open_price,sl,tp,close_time,reason,commission,swaps,close_price,profit,server_id,equity_after,p_hat,b_hat,kelly_f_star_used\n6,56113294,7400453633,USDCHF,0,2.1763,982562340,1.668,1.662,1.676,982639680,1,-0.3,-0.22,1.6763,1805.81,1,10000.02,0.593366,0.969754,0.17405\n6,56113295,7400453633,USDCHF,0,2.9008,982639740,1.677,1.671,1.685,982646640,1,-0.41,-0.29,1.685,2319.97,1,10000.04,0.613698,0.969754,0.215347\n6,56113296,7400453633,USDCHF,0,3.5891,982646700,1.6848,1.6788,1.6928,982653900,1,-0.5,-0.36,1.6932,3014.01,1,10000.07,0.633013,0.969754,0.25\n6,56113297,7400453633,USDCHF,0,4.1667,982653960,1.6941,1.6881,1.7021,982666740,0,-0.58,-0.42,1.6876,-2709.35,1,10000.04,0.601362,3.613962,0.25\n6,56113298,7400453633,USDCHF,0,4.1667,982666800,1.6885,1.6825,1.6965,982728000,0,-0.58,-0.42,1.6817,-2834.35,1,10000.02,0.571294,1.810462,0.25\n6,56113299,7400453633,USDCHF,0,4.1667,982728120,1.6805,1.6745,1.6885,982746720,0,-0.58,-0.42,1.6742,-2626.0,1,9999.98,0.542729,1.261164,0.180151\n6,56113300,7400453633,USDCHF,0,3.0025,982746780,1.6741,1.6681,1.6821,982753140,1,-0.42,-0.3,1.6821,2401.29,1,10000.0,0.565593,1.627607,0.25\n6,56113301,7400453633,USDCHF,0,4.1667,982753200,1.6818,1.6758,1.6898,982756260,1,-0.58,-0.42,1.6902,3499.0,1,10000.04,0.587313,1.997203,0.25\n6,56113302,7400453633,USDCHF,0,4.1667,982756320,1.691,1.685,1.699,982758660,0,-0.58,-0.42,1.6847,-2626.01,1,10000.01,0.557948,1.550344,0.25\n6,56113303,7400453633,USDCHF,0,4.1667,982758720,1.6847,1.6787,1.6927,982806420,1,-0.58,-0.42,1.6927,3332.34,1,10000.04,0.58005,1.80623,0.25\n"
Output:
kelly

[Example #2]
Input data (trading features in JSON):
  "records_head10": "broker_id,ticket,login,symbol,cmd,volume,open_time,open_price,sl,tp,close_time,reason,commission,swaps,close_price,profit,server_id\n6,42812938,5976650770,EURCHF,0,1,1358710020,1.2429,1.2389,1.2469,1358827980,0,-0.14,-0.1,1.23774,-51.6,1\n6,42812939,5976650770,EURCHF,0,2,1358828040,1.23771,1.23371,1.24171,1358937840,0,-0.28,-0.2,1.23321,-90.0,1\n6,42812940,5976650770,EURCHF,0,4,1358937900,1.23391,1.22991,1.23791,1358952600,1,-0.56,-0.4,1.23799,163.2,1\n6,42812941,5976650770,EURCHF,0,1,1358952660,1.23799,1.23399,1.24199,1359016080,1,-0.14,-0.1,1.24205,40.6,1\n6,42812942,5976650770,EURCHF,0,1,1359016140,1.24207,1.23807,1.24607,1359079560,1,-0.14,-0.1,1.24614,40.7,1\n6,42812943,5976650770,EURCHF,0,1,1359079620,1.24575,1.24175,1.24975,1359091920,0,-0.14,-0.1,1.24123,-45.2,1\n6,42812944,5976650770,EURCHF,0,2,1359091980,1.24114,1.23714,1.24514,1359100380,1,-0.28,-0.2,1.24514,80.0,1\n6,42812945,5976650770,EURCHF,0,1,1359100440,1.24522,1.24122,1.24922,1359117420,1,-0.14,-0.1,1.24928,40.6,1\n6,42812946,5976650770,EURCHF,0,1,1359117480,1.24921,1.24521,1.25321,1359424140,0,-0.14,-0.1,1.24439,-48.2,1\n6,42812947,5976650770,EURCHF,0,2,1359424200,1.24411,1.24011,1.24811,1359531900,0,-0.28,-0.2,1.24004,-81.4,1\n"
Output:
martin

[Example #3]
Input data (trading features in JSON):
  "records_head10": "broker_id,ticket,login,symbol,cmd,volume,open_time,open_price,sl,tp,close_time,reason,commission,swaps,close_price,profit,server_id\n6,35503520,774761,USDJPY.i,1,1,1585013880,110.923,110.496,0.0,1585024884,0,-0.07,0.0,110.499,3.84,1\n6,35504106,774761,USDJPY.i,1,1,1585014120,110.904,111.143,0.0,1585015301,0,-0.07,0.0,111.148,-2.2,1\n6,35504526,774761,USDJPY.i,1,1,1585013867,110.933,110.54,0.0,1585031154,0,-0.07,0.0,110.478,4.12,1\n6,35506755,774761,USDJPY.i,1,1,1585019034,110.889,110.78,0.0,1585020911,0,-0.07,0.0,110.789,0.9,1\n6,35506798,774761,USDJPY.i,1,1,1585019099,110.865,110.738,0.0,1585020907,0,-0.07,0.0,110.755,0.99,1\n6,35508607,774761,USDJPY.i,1,1,1585021779,110.612,110.458,0.0,1585024236,0,-0.07,0.0,110.459,1.39,1\n6,35509410,774761,USDJPY.i,1,1,1585022324,110.495,110.437,0.0,1585023285,0,-0.07,0.0,110.437,0.53,1\n6,35512317,774761,USDJPY.i,1,1,1585024277,110.404,110.52,0.0,1585024887,0,-0.07,0.0,110.519,-1.04,1\n6,35518003,774761,USDJPY.i,1,1,1585033001,110.29,110.281,0.0,1585034417,0,-0.07,0.0,110.293,-0.03,1\n6,35549452,774761,USDJPY.i,1,1,1585064904,110.768,110.759,110.52,1585066956,0,-0.07,0.0,110.759,0.08,1\n"
Output:
other

[Example #4]
Input data (trading features in JSON):
  "records_head10": "broker_id,ticket,login,symbol,cmd,volume,open_time,open_price,sl,tp,close_time,reason,commission,swaps,close_price,profit,server_id,equity_after,p_hat,b_hat,kelly_f_star_used\n6,59463511,4340406656,XAUUSD,0,1.2083,1278266460,1211.55,1211.544,1211.558,1278266520,1,-0.17,-0.12,1211.85,36249.81,1,10000.54,0.510333,1.25088,0.118875\n6,59463512,4340406656,XAUUSD,0,1.9814,1278266580,1211.85,1211.844,1211.858,1278266640,1,-0.28,-0.2,1212.0,29720.48,1,10000.82,0.534816,1.25088,0.162931\n6,59463513,4340406656,XAUUSD,0,2.7158,1278266700,1212.2,1212.194,1212.208,1278266760,1,-0.38,-0.27,1212.5,81473.39,1,10001.76,0.558076,1.25088,0.204785\n6,59463514,4340406656,XAUUSD,0,3.4136,1278266820,1212.3,1212.294,1212.308,1278266880,1,-0.48,-0.34,1212.35,17067.03,1,10001.5,0.580172,1.25088,0.244546\n6,59463515,4340406656,XAUUSD,0,4.0763,1278266940,1212.25,1212.244,1212.258,1278267000,1,-0.57,-0.41,1212.35,40762.03,1,10001.53,0.601163,1.25088,0.25\n6,59463516,4340406656,XAUUSD,0,4.1674,1278267120,1212.8,1212.794,1212.808,1278267180,0,-0.58,-0.42,1212.65,-62512.61,1,10001.32,0.571105,5.274721,0.25\n6,59463517,4340406656,XAUUSD,0,4.1673,1278267240,1212.7,1212.694,1212.708,1278267300,0,-0.58,-0.42,1212.65,-20837.51,1,10001.4,0.54255,4.110172,0.25\n6,59463518,4340406656,XAUUSD,0,4.1673,1278267420,1212.68,1212.674,1212.688,1278267480,1,-0.58,-0.42,1212.7,8333.67,1,10001.61,0.565422,4.00856,0.25\n6,59463519,4340406656,XAUUSD,0,4.1674,1278267600,1212.75,1212.744,1212.758,1278267660,0,-0.58,-0.42,1212.6,-62512.35,1,10001.48,0.537151,2.318131,0.25\n6,59463520,4340406656,XAUUSD,0,4.1672,1278267720,1212.55,1212.544,1212.558,1278267780,0,-0.58,-0.42,1212.4,-62509.74,1,10000.65,0.510294,1.655075,0.214412\n"
Output:
kelly

[Example #5]
Input data (trading features in JSON):
  "records_head10": "broker_id,ticket,login,symbol,cmd,volume,open_time,open_price,sl,tp,close_time,reason,commission,swaps,close_price,profit,server_id\n6,42812938,8292708674,USDTRY,0,1,1451840400,2.91498,2.91198,2.91898,1451840700,1,-0.14,-0.1,2.91975,47.7,1\n6,42812939,8292708674,USDTRY,0,1,1451840760,2.91948,2.91648,2.92348,1451841060,1,-0.14,-0.1,2.92377,42.9,1\n6,42812940,8292708674,USDTRY,0,1,1451841120,2.918,2.915,2.922,1451841180,0,-0.14,-0.1,2.91499,-30.1,1\n6,42812941,8292708674,USDTRY,0,2,1451841300,2.91397,2.91097,2.91797,1451843880,1,-0.28,-0.2,2.9195,110.6,1\n6,42812942,8292708674,USDTRY,0,1,1451843940,2.91806,2.91506,2.92206,1451851680,1,-0.14,-0.1,2.92305,49.9,1\n6,42812943,8292708674,USDTRY,0,1,1451851740,2.92329,2.92029,2.92729,1451854260,1,-0.14,-0.1,2.92785,45.6,1\n6,42812944,8292708674,USDTRY,0,1,1451854320,2.92615,2.92315,2.93015,1451866920,1,-0.14,-0.1,2.93041,42.6,1\n6,42812945,8292708674,USDTRY,0,1,1451866980,2.93035,2.92735,2.93435,1451867280,0,-0.14,-0.1,2.92709,-32.6,1\n6,42812946,8292708674,USDTRY,0,2,1451867340,2.92714,2.92414,2.93114,1451869740,1,-0.28,-0.2,2.93198,96.8,1\n6,42812947,8292708674,USDTRY,0,1,1451869800,2.93226,2.92926,2.93626,1451870340,1,-0.14,-0.1,2.9364,41.4,1\n"
Output:
martin

[Example #6]
Input data (trading features in JSON):
  "records_head10": "broker_id,ticket,login,symbol,cmd,volume,open_time,open_price,sl,tp,close_time,reason,commission,swaps,close_price,profit,server_id\n6,67857556,7055255,AUDCAD,0,5,1705428615,0.88846,0.0,0.0,1705586308,1,0.0,-0.63,0.88599,-9.15,1\n6,67875340,7055255,AUDCAD,0,7,1705475372,0.88638,0.0,0.0,1705586308,1,0.0,-0.66,0.88599,-2.02,1\n6,67883986,7055255,AUDCAD,0,8,1705489704,0.88417,0.0,0.0,1705586308,1,0.0,-0.75,0.88598,10.73,1\n6,67913327,7055255,AUDCAD,0,11,1705545016,0.88175,0.0,0.0,1705586309,1,0.0,0.0,0.88598,34.49,1\n6,67939055,7055255,AUDCAD,0,5,1705594631,0.88523,0.0,0.0,1705599676,1,0.0,0.0,0.88625,3.78,1\n6,69181858,7055255,AUDCAD,1,3,1708494301,0.88733,0.0,0.0,1708534570,1,0.0,0.0,0.88584,3.3,1\n6,69228756,7055255,AUDCAD,0,3,1708575301,0.88322,0.0,0.0,1708577927,1,0.0,0.0,0.8834,0.4,1\n6,69725156,7055255,AUDCAD,1,3,1709778905,0.88778,0.0,0.0,1710234244,1,0.0,0.03,0.8908,-6.72,1\n6,69735752,7055255,AUDCAD,1,4,1709805326,0.88986,0.0,0.0,1710234245,1,0.0,0.03,0.8908,-2.79,1\n6,69749083,7055255,AUDCAD,1,5,1709825410,0.89212,0.0,0.0,1710234246,1,0.0,0.03,0.89081,4.86,1\n"
Output:
other

Input data (trading features in JSON):
{features}""",
                    },
                ],
            )
            # 提取 message.content
            content = completion.choices[0].message.content
            return content
        except openai.RateLimitError as e:
            # 获取建议等待时间，否则默认2秒
            import re
            import time

            msg = str(e)
            wait_time = 2
            match = re.search(r"try again in ([0-9.]+)s", msg)
            if match:
                wait_time = float(match.group(1)) + 0.5  # 稍微多等一点
            print(f"RateLimitError: 等待{wait_time}秒后重试...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"LLM调用异常: {e}")
            raise


# 这个实验的主要目的是通过zero shot测试, 确认模型对于马丁格尔凯莉和其他交易策略的识别能力
# 实验会在qwen-max，qwen-turbo，qwq-32b，qwen3-235b-a22b上进行

modules_json_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modules.json"
)
with open(modules_json_path, "r", encoding="utf-8") as f:
    experiment_models_list = json.load(f)  # modules are important

# experiment_models_list = experiment_models_list[-1:]

if __name__ == "__main__":
    test_data_dir = os.path.join(os.path.dirname(__file__), "../test_data")
    files = os.listdir(test_data_dir)

    categories = defaultdict(list)
    for filename in tqdm(files, desc="Extracting features"):
        if os.path.isfile(os.path.join(test_data_dir, filename)):
            # 分类依据为文件名中第一个'-'之前的字符串
            category = filename.split(sep="—")[0]
            file_path = os.path.join(test_data_dir, filename)
            features = extract_features(file_path)
            categories[category].append({"filename": filename, "features": features})

    # print(categories["kelly"][0]["features"])

    for module in experiment_models_list:
        module_name, module_key, module_url = (
            module["model_name"],
            module["apikey"],
            module["baseurl"],
        )
        print(f"开始处理模型{module_name}")

        client = openai.OpenAI(
            api_key=module_key,
            base_url=module_url,
        )

        # 初始化混淆矩阵
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for category in categories:
            print(f"开始处理类别{category}")
            for item in tqdm(categories[category], desc=f"Processing {category}"):
                features = item["features"]
                res = io_llm(client, features, module_name)
                res = normalize_label(res)
                item["llm_result"] = res
                print(f"文件{item['filename']}的分类结果是: {res}")
                # 更新混淆矩阵
                confusion_matrix[category][res] += 1

        # 打印混淆矩阵
        print("混淆矩阵：")
        labels = sorted(
            confusion_matrix.keys() | {k for v in confusion_matrix.values() for k in v}
        )
        print("\t" + "\t".join(labels))
        for true_label in labels:
            row = [
                str(confusion_matrix[true_label][pred_label]) for pred_label in labels
            ]
            print(f"{true_label}\t" + "\t".join(row))

        # 保存混淆矩阵到csv（保存到和 py 文件相同的文件夹）
        ts = int(time.time())
        csv_filename = os.path.join(
            os.path.dirname(__file__), f"res_6k_{module_name}_{ts}.csv"
        )
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label"] + labels)
            for true_label in labels:
                row = [
                    confusion_matrix[true_label][pred_label] for pred_label in labels
                ]
                writer.writerow([true_label] + row)
        print(f"混淆矩阵已保存到: {csv_filename}")
