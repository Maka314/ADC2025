import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score

def read_confusion_matrix_from_csv(csv_path):
	# 读取csv，跳过前两行（表头和分隔线）
	# 跳过前两行，strip每列名和每行名，去除空列
	df = pd.read_csv(csv_path, sep='|', skiprows=2, engine='python')
	df = df.rename(columns=lambda x: x.strip())
	df = df.dropna(axis=1, how='all')
	df = df.dropna(axis=0, how='all')
	# 处理行名
	if df.columns[0] != '':
		row_label_col = df.columns[0]
	else:
		row_label_col = df.columns[1]
	row_labels = [str(i).strip().replace('**','').replace('*','') for i in df[row_label_col]]
	df.index = row_labels
	df = df.drop(row_label_col, axis=1)
	df.columns = [c.strip() for c in df.columns]
	return df

def calc_f1_from_confusion_matrix(df):
	labels = df.columns.tolist()
	y_true = []
	y_pred = []
	for i, true_label in enumerate(df.index):
		for j, pred_label in enumerate(labels):
			count = int(df.iloc[i, j])
			y_true += [true_label] * count
			y_pred += [pred_label] * count
	report = classification_report(y_true, y_pred, labels=labels, digits=3)
	macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro')
	return report, macro_f1

if __name__ == '__main__':
	import os
	csv_path = os.path.join(os.path.dirname(__file__), 'res_3k_qwen-max_1758645697_fullfeature.csv')
	df = read_confusion_matrix_from_csv(csv_path)
	report, macro_f1 = calc_f1_from_confusion_matrix(df)
	print('分类报告:')
	print(report)
	print(f'全体宏平均F1分数: {macro_f1:.3f}')
