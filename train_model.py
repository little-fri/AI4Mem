import argparse
import os

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib


def load_uvm_only(path="total_data.csv"):
    """手动解析，只保留 UVM 行，避免 KERNEL 额外逗号破坏列对齐。"""
    rows = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith('#'):
                continue
            if not line.startswith('UVM'):
                # 跳过 KERNEL 等
                continue
            parts = line.strip().split(',', 7)
            # 填充或截断到 8 列
            parts += [''] * (8 - len(parts))
            parts = parts[:8]
            rows.append(parts)
    if not rows:
        return pd.DataFrame(columns=['Category','EventName','Address','StartTime_ns','EndTime_ns','Duration_ns','Host_Phys_Info','Mapped_File'])
    df = pd.DataFrame(rows, columns=['Category','EventName','Address','StartTime_ns','EndTime_ns','Duration_ns','Host_Phys_Info','Mapped_File'])
    return df


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM prefetch model (optional继续训练)")
    parser.add_argument('--total-path', default='total_data.csv', help='总数据 CSV 路径')
    parser.add_argument('--model-path', default='uvm_prefetch_model.txt', help='模型保存/加载路径')
    parser.add_argument('--feature-path', default='feature_cols.pkl', help='特征列保存路径')
    parser.add_argument('--resume', action='store_true', help='若存在旧模型，则在其基础上继续训练')
    parser.add_argument('--num-boost-round', type=int, default=400, help='最大迭代轮次')
    parser.add_argument('--early-stop', type=int, default=30, help='提前停止轮次')
    args = parser.parse_args()

    df = load_uvm_only(args.total_path)
    if df.empty:
        print("total_data.csv 为空，无法训练")
        return

    # 规范时间戳为整数
    df['StartTime_ns'] = pd.to_numeric(df['StartTime_ns'], errors='coerce').fillna(0).astype(np.int64)
    df['EndTime_ns'] = pd.to_numeric(df['EndTime_ns'], errors='coerce').fillna(0).astype(np.int64)

    uvm_events = df.copy()
    if uvm_events.empty:
        print('没有 UVM 事件，无法训练')
        return

    # 无法可靠匹配 KERNEL（已被跳过），标记为 UNK
    uvm_events['KernelName'] = 'UNK'

    # 标签：GPU_PAGE_FAULT 为 1，其余 0
    uvm_events['label'] = (uvm_events['EventName'] == 'GPU_PAGE_FAULT').astype(int)

    # 特征工程
    def addr_to_int(x):
        try:
            return int(str(x), 16)
        except Exception:
            return 0

    uvm_events['PageAddrInt'] = uvm_events['Address'].apply(addr_to_int)
    uvm_events['KernelID'] = uvm_events['KernelName'].astype('category').cat.codes
    uvm_events['TimeSincePrevFault'] = uvm_events.groupby('Address')['StartTime_ns'].diff().fillna(0)

    feature_cols = ['PageAddrInt','KernelID','TimeSincePrevFault']
    X = uvm_events[feature_cols]
    y = uvm_events['label']

    # 按时间排序并均分 8 份：前 7 份训练，最后 1 份验证
    X = X.copy()
    X.loc[:, 'StartTime_ns'] = uvm_events['StartTime_ns'].values
    X_sorted = X.sort_values('StartTime_ns').reset_index(drop=True)
    y_sorted = y.loc[X_sorted.index].reset_index(drop=True)
    n = len(X_sorted)
    if n < 8:
        print('数据太少，无法按 8 份切分')
        return
    split_idx = n * 7 // 8
    X_train, X_val = X_sorted.iloc[:split_idx].drop(columns=['StartTime_ns']), X_sorted.iloc[split_idx:].drop(columns=['StartTime_ns'])
    y_train, y_val = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'num_threads': 4
    }

    init_model = None
    if args.resume and os.path.exists(args.model_path):
        try:
            init_model = lgb.Booster(model_file=args.model_path)
            print(f"加载已有模型继续训练: {args.model_path}")
        except Exception as e:
            print(f"加载旧模型失败，改为重新训练: {e}")
            init_model = None

    bst = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['val'],
        num_boost_round=args.num_boost_round,
        callbacks=[lgb.early_stopping(args.early_stop)],
        init_model=init_model
    )

    # 验证集简单指标
    y_val_pred = bst.predict(X_val)
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    try:
        auc = roc_auc_score(y_val, y_val_pred)
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(y_val, y_val_pred)
    except Exception:
        ap = float('nan')
    try:
        ll = log_loss(y_val, y_val_pred, eps=1e-7)
    except Exception:
        ll = float('nan')
    print(f"验证集: AUC={auc:.4f} AP={ap:.4f} LogLoss={ll:.4f} (基于最后 1/8 数据)")

    bst.save_model(args.model_path)
    joblib.dump(feature_cols, args.feature_path)
    print("训练完成，模型已保存。")


if __name__ == '__main__':
    main()
