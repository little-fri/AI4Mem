import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import utils # 确保导入了这个，因为要加载 encoder

# === 配置 ===
CACHE_DIR = 'data_cache'
DATA_FILE = os.path.join(CACHE_DIR, 'processed_data.pkl')
MODEL_FILE = os.path.join(CACHE_DIR, 'lstm_model.pth')
OUTPUT_CSV = 'predictions.csv'
ADDR_ENC_PATH = os.path.join(CACHE_DIR, 'addr_encoder.pkl')

PREDICT_STEPS = 20
EMBED_DIM = 32
HIDDEN_DIM = 64
CONFIDENCE_THRESHOLD = 0.0

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

# === 新增：从 train_model.py 搬来的智能加载函数 ===
def smart_load_weights(model, state_dict):
    """
    允许将小模型的权重加载到大模型中 (In-Memory Expansion)
    """
    model_dict = model.state_dict()
    loaded_count = 0
    
    for name, param in state_dict.items():
        if name not in model_dict:
            continue
            
        current_param = model_dict[name]
        
        # 1. 形状完全匹配
        if param.shape == current_param.shape:
            model_dict[name].copy_(param)
            loaded_count += 1
            
        # 2. Embedding 层扩容 (旧权重 -> 新权重)
        elif 'embed.weight' in name and param.shape[1] == current_param.shape[1]:
            old_rows = param.shape[0]
            new_rows = current_param.shape[0]
            if new_rows > old_rows:
                # 只复制旧的部分，新多出来的部分保持随机初始化
                model_dict[name][:old_rows, :].copy_(param)
                loaded_count += 1
                
        # 3. FC 层扩容
        elif 'fc.' in name:
            if param.shape[0] < current_param.shape[0]:
                old_out = param.shape[0]
                if 'weight' in name:
                    model_dict[name][:old_out, :].copy_(param)
                else:
                    model_dict[name][:old_out].copy_(param)
                loaded_count += 1

    model.load_state_dict(model_dict)
    return loaded_count

def infer():
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        return

    # 1. 加载数据包 (获取最新的词表大小)
    data_pkg = joblib.load(DATA_FILE)
    ctx = data_pkg['infer_context']
    if ctx is None: return
    
    # 获取当前数据的 Vocab (可能比模型大)
    data_vocab = data_pkg['vocab_sizes']
    
    # 2. 加载地址解码器
    addr_encoder = utils.IncrementalLabelEncoder().load(ADDR_ENC_PATH)

    # 3. 实例化模型 (关键修改：强制使用数据最新的 Vocab 大小)
    # 这样模型在内存里就是大的，足以容纳新 ID
    model = HotPageLSTM(data_vocab['num_kernels'], data_vocab['num_events'], data_vocab['num_pages'])
    
    # 4. 加载旧权重 (关键修改：使用智能加载)
    try:
        checkpoint = torch.load(MODEL_FILE)
        # 如果模型文件里的 vocab 和现在的 vocab 不一样，会有提示
        saved_vocab = checkpoint['vocab_sizes']
        
        if saved_vocab != data_vocab:
            print(f"Adapting model in memory: {saved_vocab} -> {data_vocab}")
            smart_load_weights(model, checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    model.eval()

    # 5. 预测
    curr_kernels = list(ctx['kernels'])
    curr_events = list(ctx['events'])
    curr_deltas = list(ctx['deltas'])
    curr_pages = list(ctx['pages'])
    last_ts = ctx['last_timestamp']
    
    results = []

    print(f"Predicting (Vocab: {data_vocab['num_pages']})...")
    with torch.no_grad():
        for i in range(PREDICT_STEPS):
            inp = np.stack([curr_kernels, curr_events, curr_deltas, curr_pages], axis=1)
            inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            
            logits = model(inp_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_id].item()
            
            # 解码
            pred_pfn = addr_encoder.inverse_transform([pred_id])[0]
            if pred_pfn != "Unknown":
                pred_addr = hex(pred_pfn << 12)
            else:
                pred_addr = "Unknown"

            last_ts += 10000
            
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