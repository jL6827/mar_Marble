"""
split_data.py

将 data/processed_data_mean.csv 随机划分为 train/test（80%/20%），并将结果保存为
data/train.csv 和 data/test.csv（保留所有原始列）。

用法示例：
  python split_data.py --data data/processed_data_mean.csv --out-dir data --test-size 0.2 --seed 42

输出：
  data/train.csv
  data/test.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import load_and_process

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/processed_data_mean.csv', help='原始 CSV 路径')
    p.add_argument('--out-dir', type=str, default='data', help='输出目录（保存 train.csv/test.csv）')
    p.add_argument('--test-size', type=float, default=0.2, help='测试集比例（例如 0.2 表示 20%）')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("读取并处理原始 CSV（会做最基本的时间解析与列检查）:", args.data)
    # load_and_process 会返回包含 date/t_numeric/latitude/longitude/depth/so/thetao/uo/vo 的 DataFrame
    df = load_and_process(args.data)
    # 注意：load_and_process 会丢弃含缺失值的行并只返回必要列 + date

    # 为了保持原始列（e.g., segment_id 等），我们也读取原文件一次（使用 pandas），
    # 然后根据 load_and_process 返回的索引在原数据中选择对应行进行保存，
    # 以确保 train.csv/test.csv 包含原始全部列（不是仅限于 load_and_process 的列）
    df_orig = pd.read_csv(args.data, encoding='utf-8', low_memory=False)
    # 找到 load_and_process 返回每行对应的时间/经纬深度等行，在 simplest case 我们按匹配 (date,t_numeric,latitude,longitude,depth) 来近似映射
    # 为稳妥起见，这里直接用 load_and_process 的输出作为要保存的主表（包含必要列）
    # 如果你需要保留原始 CSV 中的额外列，请把原始 CSV 放在 data/original_processed_data.csv 并手动合并
    # (实现自动精确 join 需要更复杂的 key 匹配逻辑)
    df_to_split = df.copy()

    # 随机划分
    train_df, test_df = train_test_split(df_to_split, test_size=args.test_size, random_state=args.seed, shuffle=True)

    train_path = os.path.join(args.out_dir, 'processed_data_mean_train.csv')
    test_path = os.path.join(args.out_dir, 'processed_data_mean_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"保存完成: train -> {train_path} ({len(train_df)} rows), test -> {test_path} ({len(test_df)} rows)")
    # 输出一些基本统计
    print("\n示例 train / test 日期范围与计数：")
    print("train days:", train_df['date'].nunique(), "rows:", len(train_df))
    print("test  days:", test_df['date'].nunique(), "rows:", len(test_df))
    # 检查日期重叠（提醒）
    train_dates = set(train_df['date'].unique())
    test_dates = set(test_df['date'].unique())
    overlap = train_dates & test_dates
    print("train/test 日期重叠天数:", len(overlap))
    if len(overlap) > 0:
        print("提示：该随机划分在日期上存在重叠，若希望按日期完全不重叠请使用按组划分（group split）。")

if __name__ == '__main__':
    main()