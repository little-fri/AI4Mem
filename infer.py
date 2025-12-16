#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import utils

# ================= 配置 =================
CACHE_DIR = 'data_cache'
DATA_FILE = os.path.join(CACHE_DIR, 'processed_data.pkl')
MODEL_FILE = os.path.join(CACHE_DIR, 'lstm_model.pth')
ADDR_ENC_PATH = os.path.join(CACHE_DIR, 'addr_encoder.pkl')

OUTPUT_CSV = 'predictions.csv'

TOPK = 4                    # 一次给几个候选页
CONF_THRESHOLD = 0.05       # 太小的概率直接丢
PREFETCH_STRIDE = 1         # 每个预测页，额外预取 ±stride 页
TIMESTAMP_DELTA_NS = 10000  # 预测时间推进

EMBED_DIM = 32
HIDDEN_DIM = 64

# ================= 模型定义 =================
class HotPageLSTM(nn.Module):
    def __init__(self, num_kernels, num_events, num_pages):
        super().__init__()
        self.kernel_embed = nn.Embedding(num_kernels, EMBED_DIM)
        self.event_embed = nn.Embedding(num_events, EMBED_DIM // 2)
        self.page_embed = nn.Embedding(num_pages, EMBED_DIM)

        input_dim = EMBED_DIM + (EMBED_DIM // 2) + EMBED_DIM + 1
        self.lstm = nn.LSTM(input_dim, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, num_pages)

    def forward(self, x):
        k = x[:, :, 0].long()
        e = x[:, :, 1].long()
        d = x[:, :, 2].unsqueeze(-1)
        p = x[:, :, 3].long()

        emb = torch.cat([
            self.kernel_embed(k),
            self.event_embed(e),
            self.page_embed(p),
            d
        ], dim=2)

        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])

# ================= 智能加载（你原来的） =================
def smart_load_weights(model, state_dict):
    model_dict = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_dict:
            continue
        cur = model_dict[name]
        if param.shape == cur.shape:
            cur.copy_(param)
        elif 'embed.weight' in name and param.shape[1] == cur.shape[1]:
            cur[:param.shape[0]].copy_(param)
        elif 'fc.' in name and param.shape[0] <= cur.shape[0]:
            if 'weight' in name:
                cur[:param.shape[0], :].copy_(param)
            else:
                cur[:param.shape[0]].copy_(param)
    model.load_state_dict(model_dict)

# ================= 主推理逻辑 =================
def infer():
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        print("Missing data or model file")
        return

    # 1. 载入数据
    data_pkg = joblib.load(DATA_FILE)
    ctx = data_pkg.get('infer_context')
    if ctx is None:
        print("No infer context")
        return

    vocab = data_pkg['vocab_sizes']

    # 2. 编码器
    addr_encoder = utils.IncrementalLabelEncoder().load(ADDR_ENC_PATH)

    # 3. 模型
    model = HotPageLSTM(
        vocab['num_kernels'],
        vocab['num_events'],
        vocab['num_pages']
    )

    ckpt = torch.load(MODEL_FILE)
    smart_load_weights(model, ckpt['model_state_dict'])
    model.eval()

    # 4. 构造输入（注意：不自回归）
    curr_k = list(ctx['kernels'])
    curr_e = list(ctx['events'])
    curr_d = list(ctx['deltas'])
    curr_p = list(ctx['pages'])
    last_ts = ctx['last_timestamp']

    inp = np.stack([curr_k, curr_e, curr_d, curr_p], axis=1)
    inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)

    results = []

    with torch.no_grad():
        logits = model(inp_tensor)
        probs = torch.softmax(logits, dim=1)

        topk = torch.topk(probs, k=min(TOPK, probs.shape[1]), dim=1)

        for pid, conf in zip(topk.indices[0], topk.values[0]):
            conf = conf.item()
            if conf < CONF_THRESHOLD:
                continue

            pid = pid.item()
            pfn = addr_encoder.inverse_transform([pid])[0]
            if pfn == "Unknown":
                continue

            base_addr = pfn << 12

            # 主预测页
            results.append({
                'Timestamp_ns': int(last_ts + TIMESTAMP_DELTA_NS),
                'Prefetch_Address': hex(base_addr),
                'Confidence': f"{conf:.3f}"
            })

            # 邻近页扩散（极其重要）
            for off in range(1, PREFETCH_STRIDE + 1):
                for sign in (-1, 1):
                    neigh = base_addr + sign * off * 4096
                    results.append({
                        'Timestamp_ns': int(last_ts + TIMESTAMP_DELTA_NS),
                        'Prefetch_Address': hex(neigh),
                        'Confidence': f"{conf * 0.5:.3f}"
                    })

    # 5. 输出
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"[Infer] Generated {len(results)} prefetch entries")
    else:
        pd.DataFrame(columns=['Timestamp_ns','Prefetch_Address','Confidence']) \
          .to_csv(OUTPUT_CSV, index=False)
        print("[Infer] No valid predictions")

if __name__ == "__main__":
    infer()
