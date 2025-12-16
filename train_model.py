import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os
import argparse

CACHE_DIR = 'data_cache'
DATA_FILE = os.path.join(CACHE_DIR, 'processed_data.pkl')
MODEL_FILE = os.path.join(CACHE_DIR, 'lstm_model.pth')

BATCH_SIZE = 64
EMBED_DIM = 32
HIDDEN_DIM = 64

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

def smart_load_weights(model, state_dict):
    """
    智能权重加载：支持 Embedding 和 Linear 层的自动扩展
    """
    model_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if name not in model_dict:
            continue
            
        current_param = model_dict[name]
        
        # 1. 如果形状完全一样，直接加载
        if param.shape == current_param.shape:
            model_dict[name].copy_(param)
            
        # 2. 如果是 Embedding 层且变大了 (行数增加)
        elif 'embed.weight' in name and param.shape[1] == current_param.shape[1]:
            old_rows = param.shape[0]
            new_rows = current_param.shape[0]
            if new_rows > old_rows:
                print(f"   Expanding {name}: {old_rows} -> {new_rows}")
                # 复制旧权重
                model_dict[name][:old_rows, :].copy_(param)
                # 新增的权重保持随机初始化
                
        # 3. 如果是全连接层且输出变大了 (通常是 fc.weight 和 fc.bias)
        elif 'fc.' in name:
            # 检查是否是输出维度变了 (dim 0)
            if param.shape[0] < current_param.shape[0]:
                old_out = param.shape[0]
                print(f"   Expanding {name}: {old_out} -> {current_param.shape[0]}")
                # fc.weight: [out_features, in_features]
                if 'weight' in name:
                    model_dict[name][:old_out, :].copy_(param)
                # fc.bias: [out_features]
                else:
                    model_dict[name][:old_out].copy_(param)

    model.load_state_dict(model_dict)

def train(epochs):
    if not os.path.exists(DATA_FILE): return

    print("Loading Data...")
    data_pkg = joblib.load(DATA_FILE)
    vocab = data_pkg['vocab_sizes']
    train_X = data_pkg['train_data']['X']
    train_Y = data_pkg['train_data']['Y']

    if len(train_X) == 0: return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 实例化新维度的模型
    print(f"Target Vocab: {vocab}")
    model = HotPageLSTM(vocab['num_kernels'], vocab['num_events'], vocab['num_pages']).to(device)
    
    # 2. 加载旧权重并自动扩展
    if os.path.exists(MODEL_FILE):
        print("Loading and Expanding old model weights...")
        try:
            old_state = torch.load(MODEL_FILE)
            smart_load_weights(model, old_state['model_state_dict'])
        except Exception as e:
            print(f"Weight loading failed: {e}, starting fresh.")
    
    # 3. 训练
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(train_X, train_Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training {epochs} epochs...")
    for epoch in range(epochs):
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_sizes': vocab 
    }, MODEL_FILE)
    print(f"Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()
    train(args.epoch)