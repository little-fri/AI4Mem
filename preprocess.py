import os
import re
import json
import argparse
from collections import Counter

import pandas as pd
import numpy as np
import torch
import joblib

PAGE_SIZE = 4096

def robust_parse_uvm(csv_path):
    # read lines and rebuild logical records using rsplit or field count
    with open(csv_path, 'r', errors='ignore') as f:
        lines = f.readlines()
    if not lines:
        return None
    body = lines[1:]
    records = []
    cur = ''
    EXPECTED = 8
    for raw in body:
        s = raw.rstrip('\n')
        if not s:
            continue
        if cur == '':
            cur = s
        else:
            cur += s
        # prefer rsplit to keep EventName commas
        if len(cur.rsplit(',', EXPECTED-1)) == EXPECTED:
            records.append(cur)
            cur = ''
        else:
            # also accept if simple split has >= EXPECTED
            if len(cur.split(',')) >= EXPECTED:
                records.append(cur)
                cur = ''
    if cur:
        records.append(cur)

    data = []
    for rec in records:
        if not rec.startswith('UVM'):
            continue
        if ('GPU_PAGE_FAULT' not in rec) and ('MIGRATE_HtoD' not in rec):
            continue
        parts = rec.rsplit(',', 7)
        addr = ''
        ts = ''
        if len(parts) >= 4:
            addr = parts[2].strip()
            ts = parts[3].strip()
        if not addr or not addr.startswith('0x'):
            m = re.search(r'0x[0-9a-fA-F]+', rec)
            addr = m.group(0) if m else ''
        if not ts or not ts.isdigit():
            m2 = re.search(r'\b(\d{8,})\b', rec)
            ts = m2.group(1) if m2 else ''
        if addr and ts and ts.isdigit():
            data.append({'timestamp': int(ts), 'abs_addr': int(addr,16)})

    df = pd.DataFrame(data)
    if df.empty:
        return None
    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    return df

def build_vocab(page_idxs, top_k):
    freq = Counter(page_idxs)
    most = [p for p,_ in freq.most_common(top_k)]
    page2idx = {p:i for i,p in enumerate(most)}
    return page2idx

def make_sequences(df, seq_length=10, horizon_ns=2_000_000_000, vocab_size=4096):
    base_addr = df['abs_addr'].min() & ~(PAGE_SIZE-1)
    df['page_idx'] = (df['abs_addr'] - base_addr) // PAGE_SIZE

    page2idx = build_vocab(df['page_idx'].tolist(), vocab_size)
    oov_id = len(page2idx)

    X = []
    Y = []
    timestamps = []
    addrs = []

    N = len(df)
    for i in range(0, N - seq_length):
        seq = df['page_idx'].iloc[i:i+seq_length].tolist()
        t_end = df['timestamp'].iloc[i+seq_length-1]
        future = df[(df['timestamp']>t_end) & (df['timestamp']<= t_end + horizon_ns)]['page_idx'].unique().tolist()
        if len(future) == 0:
            continue
        seq_ids = [page2idx.get(p, oov_id) for p in seq]
        y = np.zeros(len(page2idx), dtype=np.uint8)
        for p in future:
            if p in page2idx:
                y[page2idx[p]] = 1
        if y.sum() == 0:
            continue
        X.append(seq_ids)
        Y.append(y)
        timestamps.append(t_end)
        addrs.append(df['abs_addr'].iloc[i+seq_length-1])

    X = np.array(X, dtype=np.int64)
    Y = np.array(Y, dtype=np.uint8)
    meta = {'base_addr': int(base_addr), 'page_size': PAGE_SIZE, 'vocab_size': len(page2idx), 'oov_id': oov_id}
    return X, Y, np.array(timestamps), np.array(addrs), page2idx, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='uvm_monitor_result.csv')
    parser.add_argument('--out', default='train_data.pth')
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--seq', type=int, default=10)
    parser.add_argument('--horizon', type=float, default=2.0)
    args = parser.parse_args()

    path = args.input
    if not os.path.exists(path):
        print('input not found:', path); return
    df = robust_parse_uvm(path)
    if df is None or df.empty:
        print('no uvm events parsed'); return

    X, Y, ts, addrs, page2idx, meta = make_sequences(df, seq_length=args.seq, horizon_ns=int(args.horizon*1e9), vocab_size=args.vocab)
    if len(X) == 0:
        print('no training sequences generated')
    else:
        print('generated', len(X), 'samples; vocab_size=', meta['vocab_size'])

    obj = {'X': torch.tensor(X, dtype=torch.int64), 'Y': torch.tensor(Y, dtype=torch.uint8), 'timestamps': ts, 'addrs': addrs}
    torch.save(obj, args.out)
    joblib.dump(page2idx, args.out + '.page2idx.pkl')
    with open(args.out + '.meta.json','w') as f:
        json.dump(meta, f)
    print('saved', args.out)

if __name__ == '__main__':
    main()
