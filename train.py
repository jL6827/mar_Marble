"""
train.py (接受 --data-train / --data-test 的版本)

说明：
- 若提供 --data-train 和 --data-test 并且文件存在，则进入 presplit 模式（直接使用这些文件），
  从 train 文件中再随机拆出 val（按 --val-size）。
- 若未提供或文件不存在，则回退为按组（默认 date）做 GroupShuffleSplit 的逻辑。
- 支持隐式模型超参：--ff-dim, --hidden, --n-layers, --dropout
- 训练结束后（默认）用最佳 checkpoint 在 test 上重新推理并输出 predictions/eval。
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    try:
        ck = torch.load(path, map_location=map_location)
        return ck
    except Exception as e:
        # best-effort allowlist for sklearn StandardScaler
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
    p.add_argument('--data', type=str, default='data/processed_data_mean.csv', help='原始 CSV（备用）')
    p.add_argument('--data-train', type=str, default='data/processed_data_mean_train.csv', help='预分割训练 CSV（若存在则优先使用）')
    p.add_argument('--data-test', type=str, default='data/processed_data_mean_test.csv', help='预分割测试 CSV（若存在则优先使用）')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out-dir', type=str, default='output')
    p.add_argument('--use-marble', choices=['auto','force','none'], default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group-col', type=str, default='date')
    p.add_argument('--test-size', type=float, default=0.15, help='当未使用预分割时的 test_size（按组）')
    p.add_argument('--val-size', type=float, default=0.15, help='当未使用预分割时，trainval 上的 val_size（按组）；若使用预分割，则相对于 train 的 val_size')
    p.add_argument('--recompute-final', action='store_true', default=True)
    # implicit model hyperparams
    p.add_argument('--ff-dim', type=int, default=64)
    p.add_argument('--hidden', type=int, default=192)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.0)
    return p.parse_args()

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    feature_cols = ['t_numeric','latitude','longitude','depth']
    target_cols = ['so','thetao','uo','vo']

    # 首先检查是否存在预分割文件（优先使用）
    presplit = False
    if os.path.exists(args.data_train) and os.path.exists(args.data_test):
        # 防止 split 脚本误输出同名文件（train/test 写成同名）
        if os.path.abspath(args.data_train) == os.path.abspath(args.data_test):
            raise RuntimeError(f"检测到预分割文件 train/test 路径相同: {args.data_train}. 请修正 split 脚本，确保 train/test 文件名不同。")
        presplit = True

    if presplit:
        print("检测到预分割文件，启用 presplit 模式：", args.data_train, args.data_test)
        # 加载预分割 CSV（会使用 load_and_process 确保列和 t_numeric/date）
        df_train_all = load_and_process(args.data_train)
        df_test = load_and_process(args.data_test)
        # 从 train_all 中再拆出 val（相对于 train 的比例）
        if args.val_size > 0.0:
            train_df, val_df = train_test_split(df_train_all, test_size=args.val_size, random_state=args.seed, shuffle=True)
        else:
            train_df = df_train_all
            val_df = df_train_all.iloc[0:0]  # 空 val（不推荐）
    else:
        # 未发现 presplit，则使用原始主 CSV 并按组划分
        print("未检测到预分割文件，使用主 CSV 并按组做 group-based 划分:", args.data)
        df_all = load_and_process(args.data)
        groups = df_all[args.group_col].values
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        trainval_idx, test_idx = next(gss.split(df_all, groups=groups))
        df_trainval = df_all.iloc[trainval_idx].reset_index(drop=True)
        df_test = df_all.iloc[test_idx].reset_index(drop=True)
        # 在 trainval 上再按组拆出 val
        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval[args.group_col].values))
        train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
        val_df = df_trainval.iloc[val_idx].reset_index(drop=True)

    print(f"使用数据大小: train={len(train_df)}, val={len(val_df)}, test={len(df_test)}")

    # 构造 Dataset（fit scaler 只在 train 上）
    train_ds = OceanDataset(train_df, feature_cols, target_cols)
    val_ds = OceanDataset(val_df, feature_cols, target_cols, scaler_x=train_ds.scaler_x, scaler_y=train_ds.scaler_y)
    test_ds = OceanDataset(df_test, feature_cols, target_cols, scaler_x=train_ds.scaler_x, scaler_y=train_ds.scaler_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # model 构建（传入隐式模型超参）
    model_config = {'ff_dim': args.ff_dim, 'hidden': args.hidden, 'n_layers': args.n_layers, 'dropout': args.dropout}
    model_wrapper = MarbleWrapper(input_dim=len(feature_cols), output_dim=len(target_cols), config=model_config)

    if args.use_marble == 'force' and not getattr(model_wrapper, 'use_marble', False):
        raise RuntimeError("要求使用 Marble，但 MarbleWrapper 未检测到 Marble 可用。")
    if args.use_marble == 'none' and getattr(model_wrapper, 'use_marble', False):
        model = build_model(input_dim=len(feature_cols), output_dim=len(target_cols), kind='implicit', **model_config)
    else:
        model = model_wrapper

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.L1Loss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_val_mae = float('inf')
    best_ckpt_path = os.path.join(args.out_dir, 'model.pt')

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

            # 验证
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
            if len(preds) > 0:
                preds = np.vstack(preds); trues = np.vstack(trues)
                mae_overall = np.mean(np.abs(preds - trues))
                print(f"Epoch {epoch} validation MAE overall: {mae_overall:.6e}")
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
                        'presplit': presplit
                    }, best_ckpt_path)
                    print(f"Saved best model (val MAE {best_val_mae:.6e}) -> {best_ckpt_path}")
            else:
                print("Warning: validation set is empty, skipping val evaluation and checkpoint save.")

    else:
        print("Skipping training (epochs <= 0)")

    # 用最佳 checkpoint 在 test 上重新推理并保存结果
    if args.recompute_final:
        if not os.path.exists(best_ckpt_path):
            raise RuntimeError(f"No checkpoint found: {best_ckpt_path}")
        ck = safe_torch_load(best_ckpt_path, map_location=device)
        scaler_x = None; scaler_y = None
        if 'scaler_x_params' in ck and ck['scaler_x_params'] is not None:
            scaler_x = build_standard_scaler_from_params(ck['scaler_x_params'])
        if 'scaler_y_params' in ck and ck['scaler_y_params'] is not None:
            scaler_y = build_standard_scaler_from_params(ck['scaler_y_params'])
        if scaler_x is None:
            print("Warning: no scaler_x in ckpt, using train_ds.scaler_x")
            scaler_x = train_ds.scaler_x
        if scaler_y is None:
            print("Warning: no scaler_y in ckpt, using train_ds.scaler_y")
            scaler_y = train_ds.scaler_y

        final_model = None
        try:
            final_model = MarbleWrapper(input_dim=len(feature_cols), output_dim=len(target_cols), config=ck.get('model_config', model_config))
            final_model.load_state_dict(ck['model_state_dict'])
        except Exception as e:
            print("Warning: load into MarbleWrapper failed, falling back to implicit build_model. Error:", e)
            final_model = build_model(input_dim=len(feature_cols), output_dim=len(target_cols), kind='implicit', **ck.get('model_config', model_config))
            final_model.load_state_dict(ck['model_state_dict'])

        final_model = final_model.to(device).eval()

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

    print("Done. Best validation MAE:", best_val_mae)

if __name__ == '__main__':
    main()