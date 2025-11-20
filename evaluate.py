import pandas as pd
import numpy as np

def evaluate_and_save(predictions_df, per_var_path, per_day_path):
    """
    predictions_df 期望列：
      date, true_so, true_thetao, true_uo, true_vo, pred_so, pred_thetao, pred_uo, pred_vo
    函数更容错：会尝试自动检测 true_/pred_ 前缀并按变量顺序计算 MAE。
    输出：
      per_var_path -> CSV (columns: variable,mae,count)
      per_day_path -> CSV (columns: date, mae_so, mae_thetao, mae_uo, mae_vo, mae_overall)
    """
    # 自动识别变量名（尝试匹配 true_ 前缀）
    cols = predictions_df.columns.tolist()
    true_cols = [c for c in cols if c.startswith('true_')]
    pred_cols = [c for c in cols if c.startswith('pred_')]

    if not true_cols or not pred_cols:
        raise RuntimeError("predictions_df 必须包含以 'true_' 和 'pred_' 前缀命名的列，例如 true_uo, pred_uo")

    # 变量名集合（去掉前缀）
    vars_true = [c[len('true_'):] for c in true_cols]
    vars_pred = [c[len('pred_'):] for c in pred_cols]
    vars_set = [v for v in vars_true if v in vars_pred]
    if not vars_set:
        raise RuntimeError("未能在 true_ 和 pred_ 列之间找到匹配的变量名。")

    variables = vars_set  # 保持顺序为 true 列出现的顺序

    # per-var MAE (overall)
    maes = []
    for v in variables:
        tcol = f'true_{v}'
        pcol = f'pred_{v}'
        diff = np.abs(predictions_df[pcol].values - predictions_df[tcol].values)
        mae = float(np.mean(diff))
        maes.append({'variable': v, 'mae': mae, 'count': int(len(diff))})

    df_var = pd.DataFrame(maes)
    df_var.to_csv(per_var_path, index=False)

    # per-day MAE
    grouped = predictions_df.groupby('date')
    rows = []
    for date, g in grouped:
        row = {'date': date}
        per_var_vals = []
        for v in variables:
            tcol = f'true_{v}'
            pcol = f'pred_{v}'
            mae = float(np.mean(np.abs(g[pcol].values - g[tcol].values)))
            row[f'mae_{v}'] = mae
            per_var_vals.append(mae)
        row['mae_overall'] = float(np.mean(per_var_vals))
        rows.append(row)
    df_days = pd.DataFrame(rows)

    # 按日期排序（尝试解析为日期）
    try:
        df_days['date_dt'] = pd.to_datetime(df_days['date'])
        df_days = df_days.sort_values('date_dt').drop(columns=['date_dt'])
    except Exception:
        df_days = df_days.sort_values('date')

    # 确保输出列顺序为 date, mae_so, mae_thetao, mae_uo, mae_vo, mae_overall （若变量存在）
    ordered_cols = ['date']
    for v in ['so','thetao','uo','vo']:
        if f'mae_{v}' in df_days.columns:
            ordered_cols.append(f'mae_{v}')
    ordered_cols.append('mae_overall')
    df_days = df_days[ordered_cols]

    df_days.to_csv(per_day_path, index=False)
    print(f"Saved {per_var_path} and {per_day_path}")