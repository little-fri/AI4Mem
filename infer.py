import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
from heapq import nlargest

PAGE_SIZE = 4096

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, out_size=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size+1, emb_dim, padding_idx=vocab_size)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, out_size)

    def forward(self, x):
        e = self.embed(x)
        o, _ = self.lstm(e)
        h = o[:,-1,:]
        return self.fc(h)

def robust_parse_last(csv_path, seq_len=10):
    # reuse simple parsing: find UVM events and return last seq_len page_idxs and last timestamp
    import re
    lines = open(csv_path, 'r', errors='ignore').read().splitlines()
    if len(lines) < 2: return None
    body = lines[1:]
    records = []
    cur = ''
    EXPECTED = 8
    for raw in body:
        s = raw.rstrip('\n')
        if not s: continue
        if cur == '': cur = s
        else: cur += s
        if len(cur.rsplit(',', EXPECTED-1)) == EXPECTED:
            records.append(cur); cur = ''
        else:
            if len(cur.split(',')) >= EXPECTED:
                records.append(cur); cur = ''
    if cur: records.append(cur)

    events = []
    for rec in records:
        if not rec.startswith('UVM'): continue
        if ('GPU_PAGE_FAULT' not in rec) and ('MIGRATE_HtoD' not in rec): continue
        parts = rec.rsplit(',',7)
        addr=''; ts=''
        if len(parts)>=4:
            addr=parts[2].strip(); ts=parts[3].strip()
        if not addr or not addr.startswith('0x'):
            m = re.search(r'0x[0-9a-fA-F]+', rec)
            addr = m.group(0) if m else ''
        if not ts or not ts.isdigit():
            m2 = re.search(r'\b(\d{8,})\b', rec)
            ts = m2.group(1) if m2 else ''
        if addr and ts and ts.isdigit():
            events.append((int(ts), int(addr,16)))
    if not events: return None
    events = sorted(events)
    last = events[-seq_len:]
    page_idxs = [ (a - (min(e[1] for e in events) & ~(PAGE_SIZE-1))) // PAGE_SIZE for _,a in last]
    last_ts = last[-1][0]
    return page_idxs, last_ts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.pth')
    parser.add_argument('--data', default='train_data.pth')
    parser.add_argument('--csv', default='uvm_monitor_result.csv')
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--out', default='prefetch_schedule.csv')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print('model not found'); return
    ck = torch.load(args.model, map_location='cpu')
    meta = ck.get('meta') if 'meta' in ck else json.load(open(args.data + '.meta.json'))
    vocab_size = meta['vocab_size']
    base_addr = meta['base_addr']

    model = LSTMModel(vocab_size=vocab_size, out_size=vocab_size)
    model.load_state_dict(ck['model_state'])
    model.eval()

    # parse last sequence
    parsed = robust_parse_last(args.csv, seq_len=10)
    if parsed is None:
        print('no recent events'); return
    seq_page_idxs, last_ts = parsed
    # map page_idx -> id: load mapping
    page2idx = None
    p2path = args.data + '.page2idx.pkl'
    import joblib
    if os.path.exists(p2path):
        page2idx = joblib.load(p2path)
    else:
        print('page2idx not found', p2path); return
    oov = meta['oov_id']
    seq_ids = [page2idx.get(p, oov) for p in seq_page_idxs]
    import numpy as np
    xb = torch.tensor([seq_ids], dtype=torch.int64)
    with torch.no_grad():
        logits = model(xb).squeeze(0).numpy()
    probs = 1/(1+np.exp(-logits))
    # take top-k ids
    topk = np.argsort(-probs)[:args.topk]
    # invert mapping
    idx2page = {v:k for k,v in page2idx.items()}
    rows = []
    for idx in topk:
        if idx in idx2page:
            page = idx2page[idx]
            addr = base_addr + page*PAGE_SIZE
            score = float(probs[idx])
            rows.append((last_ts, hex(addr), int(page), score))

    # write CSV: trigger_time_ns,page_addr,page_idx,score
    with open(args.out,'w') as f:
        f.write('trigger_time_ns,page_addr,page_idx,score\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")
    print('wrote', args.out)

if __name__ == '__main__':
    main()
