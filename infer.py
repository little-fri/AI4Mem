import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import utils # 导入 utils

CACHE_DIR = 'data_cache'
DATA_FILE = os.path.join(CACHE_DIR, 'processed_data.pkl')
MODEL_FILE = os.path.join(CACHE_DIR, 'lstm_model.pth')
OUTPUT_CSV = 'predictions.csv'
ADDR_ENC_PATH = os.path.join(CACHE_DIR, 'addr_encoder.pkl') # 单独加载 encoder

PREDICT_STEPS = 20
EMBED_DIM = 32
HIDDEN_DIM = 64
CONFIDENCE_THRESHOLD = 0.0

# 模型定义同 Train
class HotPageLSTM(nn.Module):
    def __init__(self, num_kernels, num_events, num_pages):
        super(HotPageLSTM, self).__init__()
        self.kernel_embed = nn.Embedding(num_kernels, EMBED_DIM)
        self.event_embed = nn.Embedding(num_events, EMBED_DIM // 2)
        self.page_embed = nn.Embedding(num_pages, EMBED_DIM)
        input_dim = EMBED_DIM + (EMBED_DIM // 2) + EMBED_DIM + 1
        self.lstm = nn.LSTM(input_dim, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, num_pages)
    
    def forward(self, x):
        k_idx = x[:, :, 0].long()
        e_idx = x[:, :, 1].long()
        delta = x[:, :, 2].unsqueeze(-1)
        p_idx = x[:, :, 3].long()
        lstm_input = torch.cat([
            self.kernel_embed(k_idx), 
            self.event_embed(e_idx), 
            self.page_embed(p_idx), 
            delta
        ], dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        return self.fc(lstm_out[:, -1, :])

def infer():
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        return

    # 1. 加载数据包
    data_pkg = joblib.load(DATA_FILE)
    ctx = data_pkg['infer_context']
    if ctx is None: return
    
    # 2. 加载地址解码器
    # 注意：现在需要用 utils.IncrementalLabelEncoder 加载
    addr_encoder = utils.IncrementalLabelEncoder().load(ADDR_ENC_PATH)

    # 3. 加载模型
    checkpoint = torch.load(MODEL_FILE)
    vocab = checkpoint['vocab_sizes']
    model = HotPageLSTM(vocab['num_kernels'], vocab['num_events'], vocab['num_pages'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. 预测
    curr_kernels = list(ctx['kernels'])
    curr_events = list(ctx['events'])
    curr_deltas = list(ctx['deltas'])
    curr_pages = list(ctx['pages'])
    last_ts = ctx['last_timestamp']
    
    results = []

    print(f"Predicting (Vocab: {vocab['num_pages']})...")
    with torch.no_grad():
        for i in range(PREDICT_STEPS):
            inp = np.stack([curr_kernels, curr_events, curr_deltas, curr_pages], axis=1)
            inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            
            logits = model(inp_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_id].item()
            
            # 解码
            # incremental encoder 的 inverse_transform 返回 numpy array
            pred_pfn = addr_encoder.inverse_transform([pred_id])[0]
            if pred_pfn != "Unknown":
                pred_addr = hex(pred_pfn << 12)
            else:
                pred_addr = "Unknown"

            last_ts += 10000
            
            # print(f"  Pred: {pred_addr} ({conf:.2f})") # 调试用

            if conf >= CONFIDENCE_THRESHOLD:
                results.append({
                    'Timestamp_ns': int(last_ts),
                    'Prefetch_Address': pred_addr,
                    'Confidence': f"{conf:.2f}"
                })
            
            curr_kernels.pop(0); curr_kernels.append(curr_kernels[-1])
            curr_events.pop(0); curr_events.append(curr_events[-1])
            curr_deltas.pop(0); curr_deltas.append(np.log1p(10000))
            curr_pages.pop(0); curr_pages.append(pred_id)

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"Inference Done. Saved to {OUTPUT_CSV}")
    else:
        pd.DataFrame(columns=['Timestamp_ns','Prefetch_Address','Confidence']).to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    infer()