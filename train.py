"""
train.py

更新说明（中文）：
- 集成了基于隐式表示的模型配置（通过 model.MarbleWrapper 回退到 ImplicitCoordNet）。
- 训练时按组（默认按 date）先划出 test 集（GroupShuffleSplit），再在 trainval 上按组划分 train/val 进行训练与选模；
  这样保证测试组与训练/验证组不重叠（适合时空外推评估）。
- 支持在命令行设置隐式模型超参：--ff-dim, --hidden, --n-layers, --dropout， 会通过 config 传入 MarbleWrapper。
- 保存 checkpoint 时只保存模型 state_dict 与 scaler 参数（避免 sklearn 对象被 pickle 问题）。
- 训练结束后默认会用最佳 checkpoint 在 test 集上重新推理并输出 predictions/eval 文件（output/ 下）。
- 兼容 GPU 自动混合精度（torch.amp.autocast）。

使用示例：
  python train.py --data data/processed_data_mean.csv --epochs 50 --batch-size 512 --ff-dim 64 --hidden 256 --n-layers 3

如果你只想用已有 checkpoint 在 group-test 上生成预测（不训练）：
  python train.py --data data/processed_data_mean.csv --epochs 0 --recompute-final

"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from data import load_and_process, OceanDataset
from model import MarbleWrapper, build_model
from evaluate import evaluate_and_save

def save_scaler_params(scaler):
    try:
        return {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'var': getattr(scaler, 'var_', scaler.scale_).tolist(),
            'n_samples_seen': int(getattr(scaler, 'n_samples_seen_', 0))
        }
    except Exception:
        return None

def build_standard_scaler_from_params(params):
    from sklearn.preprocessing import StandardScaler
    import numpy as _np
    sc = StandardScaler()
    sc.mean_ = _np.array(params['mean'], dtype=_np.float64)
    sc.scale_ = _np.array(params['scale'], dtype=_np.float64)
    sc.var_ = _np.array(params.get('var', params['scale']), dtype=_np.float64)
    sc.n_samples_seen_ = int(params.get('n_samples_seen', 0))
    return sc

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    y_orig = torch.stack([b[2] for b in batch])
    dates = [b[3] for b in batch]
    return xs, ys, y_orig, dates

def safe_torch_load(path, map_location=None):
    """
    Safe wrapper to try to load older checkpoints that may contain pickled sklearn objects.
    """
    try:
        ck = torch.load(path, map_location=map_location)
        return ck
    except Exception as e:
        # try allowlist for sklearn StandardScaler then reload (best-effort)
        try:
            import torch.serialization as _ser
            try:
                import sklearn.preprocessing._data as _sk
                _ser.add_safe_globals([_sk.StandardScaler])
            except Exception:
                try:
                    import sklearn.preprocessing as _skp
                    if hasattr(_skp, 'StandardScaler'):
                        _ser.add_safe_globals([_skp.StandardScaler])
                except Exception:
                    pass
            ck = torch.load(path, map_location=map_location, weights_only=False)
            return ck
        except Exception as e2:
            raise RuntimeError(f"Failed to load checkpoint: {e}\nRetry failed: {e2}")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/processed_data_mean.csv', help='输入 CSV 路径')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out-dir', type=str, default='output')
    p.add_argument('--use-marble', choices=['auto','force','none'], default='auto',
                   help='尝试使用本地 marble（auto），force 强制要求 marble 可用，none 强制使用回退模型')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group-col', type=str, default='date', help='按哪列做 group split（默认 date）')
    p.add_argument('--test-size', type=float, default=0.15, help='最终 test 集占比（按组）')
    p.add_argument('--val-size', type=float, default=0.15, help='在 trainval 上再划出 val 的比例（按组）')
    p.add_argument('--recompute-final', action='store_true', default=True,
                   help='训练后加载最佳 checkpoint 在 test 集上重新推理并保存评估')
    # implicit model hyperparams
    p.add_argument('--ff-dim', type=int, default=64, help='Fourier feature mapping size')
    p.add_argument('--hidden', type=int, default=256, help='隐式 MLP 宽度')
    p.add_argument('--n-layers', type=int, default=3, help='隐式 MLP 层数')
    p.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    return p.parse_args()

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load data and process
    df = load_and_process(args.data)
    if args.group_col not in df.columns:
        raise RuntimeError(f"group 列 {args.group_col} 不存在于数据中，可用列：{df.columns.tolist()}")

    feature_cols = ['t_numeric','latitude','longitude','depth']
    target_cols = ['so','thetao','uo','vo']

    # 1) 按组先划出 test 集（最终保留）
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    groups = df[args.group_col].values
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 2) 在 trainval 上按组再划分 train / val（用于每 epoch 验证）
    gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval[args.group_col].values))
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    print(f"Groups: train days {len(df_train[args.group_col].unique())}, val days {len(df_val[args.group_col].unique())}, test days {len(df_test[args.group_col].unique())}")
    print(f"Rows: train {len(df_train)}, val {len(df_val)}, test {len(df_test)}")

    # datasets (fit scaler on train only)
    train_ds = OceanDataset(df_train, feature_cols, target_cols)
    val_ds = OceanDataset(df_val, feature_cols, target_cols, scaler_x=train_ds.scaler_x, scaler_y=train_ds.scaler_y)
    test_ds = OceanDataset(df_test, feature_cols, target_cols, scaler_x=train_ds.scaler_x, scaler_y=train_ds.scaler_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # build model (use MarbleWrapper; pass implicit model config so wrapper's fallback uses those)
    model_config = {'ff_dim': args.ff_dim, 'hidden': args.hidden, 'n_layers': args.n_layers, 'dropout': args.dropout}
    model_wrapper = MarbleWrapper(input_dim=len(feature_cols), output_dim=len(target_cols), config=model_config)

    if args.use_marble == 'force' and not getattr(model_wrapper, 'use_marble', False):
        raise RuntimeError("要求使用 marble，但 MarbleWrapper 未检测到可用的 marble 实现。")
    if args.use_marble == 'none' and getattr(model_wrapper, 'use_marble', False):
        # 强制回退：直接用 build_model 的 implicit
        model = build_model(input_dim=len(feature_cols), output_dim=len(target_cols), kind='implicit', ff_dim=args.ff_dim, hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout)
    else:
        model = model_wrapper

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.L1Loss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_val_mae = float('inf')
    best_ckpt_path = os.path.join(args.out_dir, 'model.pt')

    # Training loop
    if args.epochs > 0:
        for epoch in range(1, args.epochs+1):
            model.train()
            running = 0.0
            it = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for x, y_scaled, y_orig, dates in pbar:
                x = x.to(device); y_scaled = y_scaled.to(device)
                optimizer.zero_grad()
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        out_scaled = model(x)
                        loss = loss_fn(out_scaled, y_scaled)
                else:
                    out_scaled = model(x)
                    loss = loss_fn(out_scaled, y_scaled)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running += float(loss.item()); it += 1
                pbar.set_postfix({'train_loss': running/it})

            # validation
            model.eval()
            preds = []; trues = []
            with torch.no_grad():
                for x, y_scaled, y_orig, dates in val_loader:
                    x = x.to(device)
                    if device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda'):
                            out_scaled = model(x)
                    else:
                        out_scaled = model(x)
                    out_scaled = out_scaled.cpu().numpy()
                    out = val_ds.scaler_y.inverse_transform(out_scaled)
                    preds.append(out); trues.append(y_orig.numpy())
            preds = np.vstack(preds); trues = np.vstack(trues)
            mae_overall = np.mean(np.abs(preds - trues))
            print(f"Epoch {epoch} validation MAE overall: {mae_overall:.6e}")

            # save best checkpoint (save scaler params, not objects)
            if mae_overall < best_val_mae:
                best_val_mae = mae_overall
                try:
                    sxp = save_scaler_params(train_ds.scaler_x)
                    syp = save_scaler_params(train_ds.scaler_y)
                except Exception:
                    sxp = None; syp = None
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler_x_params': sxp,
                    'scaler_y_params': syp,
                    'feature_cols': feature_cols,
                    'target_cols': target_cols,
                    'model_config': model_config,
                    'seed': args.seed,
                    'group_col': args.group_col
                }, best_ckpt_path)
                print(f"Saved best model (val MAE {best_val_mae:.6e}) -> {best_ckpt_path}")
    else:
        print("跳过训练（epochs <= 0）")

    # recompute final predictions on test using best checkpoint
    if args.recompute_final:
        if not os.path.exists(best_ckpt_path):
            raise RuntimeError(f"找不到最佳 checkpoint: {best_ckpt_path}")
        print("Loading best checkpoint:", best_ckpt_path)
        ck = safe_torch_load(best_ckpt_path, map_location=device)

        # reconstruct scalers from params if available
        scaler_x = None; scaler_y = None
        if 'scaler_x_params' in ck and ck['scaler_x_params'] is not None:
            scaler_x = build_standard_scaler_from_params(ck['scaler_x_params'])
        if 'scaler_y_params' in ck and ck['scaler_y_params'] is not None:
            scaler_y = build_standard_scaler_from_params(ck['scaler_y_params'])
        if scaler_x is None:
            print("Warning: checkpoint 中没有 scaler_x_params，使用 train_ds 的 scaler_x 作为回退。")
            scaler_x = train_ds.scaler_x
        if scaler_y is None:
            print("Warning: checkpoint 中没有 scaler_y_params，使用 train_ds 的 scaler_y 作为回退。")
            scaler_y = train_ds.scaler_y

        # load model weights into a new instance (respecting MarbleWrapper)
        final_model = None
        try:
            final_model = MarbleWrapper(input_dim=len(feature_cols), output_dim=len(target_cols), config=ck.get('model_config', model_config))
            final_model.load_state_dict(ck['model_state_dict'])
            print("Loaded state into MarbleWrapper/fallback model.")
        except Exception as e:
            print("Warning: 无法将参数加载进 MarbleWrapper，尝试回退 build_model MLP。Error:", e)
            final_model = build_model(input_dim=len(feature_cols), output_dim=len(target_cols), kind='implicit', **ck.get('model_config', model_config))
            final_model.load_state_dict(ck['model_state_dict'])

        final_model = final_model.to(device).eval()

        # create test dataset with the saved scalers
        test_ds = OceanDataset(df_test, feature_cols, target_cols, scaler_x=scaler_x, scaler_y=scaler_y)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        preds = []; trues = []; dates_all = []
        with torch.no_grad():
            for x, y_scaled, y_orig, dates in test_loader:
                x = x.to(device)
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        out_scaled = final_model(x)
                else:
                    out_scaled = final_model(x)
                out_scaled = out_scaled.cpu().numpy()
                out = test_ds.scaler_y.inverse_transform(out_scaled)
                preds.append(out); trues.append(y_orig.numpy()); dates_all.extend(dates)
        preds = np.vstack(preds); trues = np.vstack(trues)

        # save predictions and eval
        preds_df = pd.DataFrame(preds, columns=target_cols)
        trues_df = pd.DataFrame(trues, columns=target_cols)
        out_df = pd.concat([pd.DataFrame({'date': dates_all}), trues_df.add_prefix('true_'), preds_df.add_prefix('pred_')], axis=1)
        preds_path = os.path.join(args.out_dir, 'predictions.csv')
        out_df.to_csv(preds_path, index=False)
        print("Saved predictions ->", preds_path)
        per_var_path = os.path.join(args.out_dir, 'eval_per_var.csv')
        per_day_path = os.path.join(args.out_dir, 'eval_per_day.csv')
        evaluate_and_save(out_df, per_var_path, per_day_path)
        print("Saved eval files:", per_var_path, per_day_path)

    print("Finished. Best validation MAE:", best_val_mae)

if __name__ == '__main__':
    main()