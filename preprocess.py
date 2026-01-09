import pandas as pd
import numpy as np
import torch
import joblib
import os
import utils # 导入刚才写的 utils.py

# === 配置 ===
INPUT_FILE = 'data.csv'
OUTPUT_DIR = 'data_cache'
DATA_PACKAGE_FILE = os.path.join(OUTPUT_DIR, 'processed_data.pkl')
SEQUENCE_LENGTH = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 路径定义
ADDR_ENC_PATH = os.path.join(OUTPUT_DIR, 'addr_encoder.pkl')
KERNEL_ENC_PATH = os.path.join(OUTPUT_DIR, 'kernel_encoder.pkl')
EVENT_ENC_PATH = os.path.join(OUTPUT_DIR, 'event_encoder.pkl')

def parse_address(addr_str):
    try:
        if isinstance(addr_str, str):
            if addr_str.strip() == '': return 0
            return int(addr_str, 16) >> 12
        return int(addr_str) >> 12
    except:
        return 0

def find_timestamp(row):
    for i in range(2, 20):
        col_name = f'Col_{i}'
        if col_name not in row: continue
        try:
            val = float(row[col_name])
            if val > 1e16: return val
        except: continue
    return np.nan

def preprocess():
    print(f"[preprocess]: 1.读取 {INPUT_FILE}]")
    try:
        df = pd.read_csv(INPUT_FILE, header=None, names=[f'Col_{i}' for i in range(30)], low_memory=False)
        if str(df.iloc[0]['Col_0']).strip() == 'Category':
            df = df.iloc[1:].copy()
        df['Category'] = df['Col_0'].astype(str).str.strip()
    except Exception as e:
        print(f"[preprocess]: Error reading CSV: {e}")
        return

    # 清洗数据
    uvm_mask = df['Category'] == 'UVM'
    df_uvm = df[uvm_mask].copy()
    df_uvm['EventName'] = df_uvm['Col_1'].str.strip()
    df_uvm['Address'] = df_uvm['Col_2']
    df_uvm['StartTime_ns'] = pd.to_numeric(df_uvm['Col_3'], errors='coerce')

    kernel_mask = df['Category'] == 'KERNEL'
    df_kernel = df[kernel_mask].copy()
    df_kernel['EventName'] = df_kernel['Col_1'].str.strip()
    df_kernel['Address'] = 0
    df_kernel['StartTime_ns'] = df_kernel.apply(find_timestamp, axis=1)

    df_final = pd.concat([df_uvm, df_kernel]).dropna(subset=['StartTime_ns'])
    df_final = df_final.sort_values('StartTime_ns').reset_index(drop=True)

    if len(df_final) == 0:
        print("[preprocess]: No valid data found.")
        return

    # 上下文填充
    df_final['CurrentKernel'] = 'IDLE'
    df_final.loc[df_final['Category'] == 'KERNEL', 'CurrentKernel'] = df_final['EventName']
    df_final['CurrentKernel'] = df_final['CurrentKernel'].replace('IDLE', np.nan).ffill().fillna('IDLE')

    target_df = df_final[df_final['Category'] == 'UVM'].copy()
    target_df['PagePFN'] = target_df['Address'].apply(parse_address)

    # === 关键修改：增量编码 ===
    print("[preprocess]: 2. 编码数据")
    
    # 1. 加载或新建 Encoder
    addr_enc = utils.IncrementalLabelEncoder().load(ADDR_ENC_PATH)
    kernel_enc = utils.IncrementalLabelEncoder().load(KERNEL_ENC_PATH)
    event_enc = utils.IncrementalLabelEncoder().load(EVENT_ENC_PATH)
    
    # 2. 增量学习 (把新出现的词加入字典)
    addr_enc.partial_fit(target_df['PagePFN'].values)
    kernel_enc.partial_fit(target_df['CurrentKernel'].values)
    event_enc.partial_fit(target_df['EventName'].values)
    
    # 3. 保存更新后的 Encoder
    addr_enc.save(ADDR_ENC_PATH)
    kernel_enc.save(KERNEL_ENC_PATH)
    event_enc.save(EVENT_ENC_PATH)
    
    # 4. 转换数据
    page_ids = addr_enc.transform(target_df['PagePFN'].values)
    kernel_ids = kernel_enc.transform(target_df['CurrentKernel'].values)
    event_ids = event_enc.transform(target_df['EventName'].values)

    print(f"[preprocess]: 当前 Vocab: Pages={len(addr_enc)}, Kernels={len(kernel_enc)}")

    # 序列构建
    target_df['TimeDelta'] = target_df['StartTime_ns'].diff().fillna(0)
    target_df['LogDelta'] = np.log1p(target_df['TimeDelta'])
    log_deltas = target_df['LogDelta'].values

    data_X, data_Y = [], []
    if len(target_df) > SEQUENCE_LENGTH:
        for i in range(len(target_df) - SEQUENCE_LENGTH):
            x_seq = np.stack([
                kernel_ids[i:i+SEQUENCE_LENGTH],
                event_ids[i:i+SEQUENCE_LENGTH],
                log_deltas[i:i+SEQUENCE_LENGTH],
                page_ids[i:i+SEQUENCE_LENGTH]
            ], axis=1)
            data_X.append(x_seq)
            data_Y.append(page_ids[i + SEQUENCE_LENGTH])

    # 准备上下文
    last_ctx = None
    if len(target_df) >= SEQUENCE_LENGTH:
        last_ctx = {
            'kernels': kernel_ids[-SEQUENCE_LENGTH:],
            'events': event_ids[-SEQUENCE_LENGTH:],
            'deltas': log_deltas[-SEQUENCE_LENGTH:],
            'pages': page_ids[-SEQUENCE_LENGTH:],
            'last_timestamp': target_df['StartTime_ns'].values[-1]
        }

    # 保存
    data_package = {
        'vocab_sizes': {
            'num_pages': len(addr_enc),
            'num_kernels': len(kernel_enc),
            'num_events': len(event_enc)
        },
        'train_data': {
            'X': torch.tensor(np.array(data_X), dtype=torch.float32) if data_X else torch.empty(0),
            'Y': torch.tensor(np.array(data_Y), dtype=torch.long) if data_Y else torch.empty(0)
        },
        'infer_context': last_ctx
    }

    joblib.dump(data_package, DATA_PACKAGE_FILE)
    print(f"[preprocess]: 3. 保存预处理数据到 {DATA_PACKAGE_FILE}")

if __name__ == "__main__":
    preprocess()