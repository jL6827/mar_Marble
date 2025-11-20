import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

class OceanDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, scaler_x=None, scaler_y=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        X = df[feature_cols].values.astype(np.float32)
        Y = df[target_cols].values.astype(np.float32)

        if scaler_x is None:
            self.scaler_x = StandardScaler()
            self.scaler_x.fit(X)
        else:
            self.scaler_x = scaler_x

        if scaler_y is None:
            self.scaler_y = StandardScaler()
            self.scaler_y.fit(Y)
        else:
            self.scaler_y = scaler_y

        self.X = self.scaler_x.transform(X).astype(np.float32)
        self.Y = self.scaler_y.transform(Y).astype(np.float32)

        # keep date strings for per-day evaluation
        if 'date' in df.columns:
            self.dates = df['date'].astype(str).values
        else:
            self.dates = np.array([''] * len(df))

        # keep original unscaled targets for evaluation convenience
        self.Y_orig = Y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        y_orig = torch.from_numpy(self.Y_orig[idx])
        date = self.dates[idx]
        return x, y, y_orig, date


def load_and_process(csv_path):
    """
    读取并标准化 CSV，返回 DataFrame，至少包含列：
      ['date','t_numeric','latitude','longitude','depth','so','thetao','uo','vo']

    兼容性与鲁棒性：
    - 自动识别 time/date 列名：time, date, datetime, timestamp
    - 尝试多种编码读取（utf-8, utf-8-sig, latin1）
    - 丢弃含必需字段缺失值的行并报出数量
    """
    csv_path = os.path.expanduser(csv_path)
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件未找到: {csv_path}. 当前工作目录: {os.getcwd()}")

    # 读取尝试：utf-8 -> utf-8-sig -> latin1
    read_err = None
    for enc in (None, 'utf-8-sig', 'latin1'):
        try:
            if enc is None:
                df = pd.read_csv(csv_path)
            else:
                df = pd.read_csv(csv_path, encoding=enc)
            read_err = None
            break
        except Exception as e:
            read_err = e
            df = None
    if df is None:
        raise RuntimeError(f"读取 CSV 失败: {read_err}")

    # 识别时间列
    time_col = None
    for c in ['time', 'date', 'datetime', 'timestamp']:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # 尝试查找列名中包含 time 或 date 的列
        for c in df.columns:
            if 'time' in c.lower() or 'date' in c.lower():
                time_col = c
                break
    if time_col is None:
        raise RuntimeError("CSV 必须包含时间列，列名应为 time/date/datetime/timestamp 之一，或包含 'time'/'date' 字样。")

    # 解析时间
    try:
        df['time_parsed'] = pd.to_datetime(df[time_col])
    except Exception:
        # 尝试 dayfirst=False fallback
        df['time_parsed'] = pd.to_datetime(df[time_col], errors='coerce')
    if df['time_parsed'].isna().any():
        nbad = int(df['time_parsed'].isna().sum())
        raise RuntimeError(f"时间列解析失败，{nbad} 行的时间无法解析，请检查格式。")

    df['date'] = df['time_parsed'].dt.strftime('%Y-%m-%d')

    # numeric time: days since start
    t0 = df['time_parsed'].min()
    df['t_numeric'] = (df['time_parsed'] - t0).dt.total_seconds() / (3600 * 24)

    # ensure lat/lon/depth exist (try alternatives)
    if 'latitude' not in df.columns:
        for alt in ['lat', 'y']:
            if alt in df.columns:
                df['latitude'] = df[alt]
                break
    if 'longitude' not in df.columns:
        for alt in ['lon', 'x']:
            if alt in df.columns:
                df['longitude'] = df[alt]
                break
    if 'depth' not in df.columns:
        for alt in ['z', 'depth_m']:
            if alt in df.columns:
                df['depth'] = df[alt]
                break

    required = ['t_numeric', 'latitude', 'longitude', 'depth', 'so', 'thetao', 'uo', 'vo']
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise RuntimeError(f"缺少必要列: {missing}. CSV 必须包含这些列（或其替代名）")

    # 丢弃含缺失值的行，防止训练报错
    before = len(df)
    df = df.dropna(subset=required).copy()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"警告: 丢弃了 {dropped} 行（含必需字段的缺失值）")

    # 最终保留所需列（并保留 date）
    out = df[['date', 't_numeric', 'latitude', 'longitude', 'depth', 'so', 'thetao', 'uo', 'vo']].copy()
    return out