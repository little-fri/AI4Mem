import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, out_size=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size+1, emb_dim, padding_idx=vocab_size)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, out_size)

    def forward(self, x):
        # x: (B, L)
        e = self.embed(x)
        o, _ = self.lstm(e)
        h = o[:,-1,:]
        return self.fc(h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='train_data.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='model.pth')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print('data not found:', args.data); return
    obj = torch.load(args.data)
    X = obj['X']
    Y = obj['Y'].float()
    if X.size(0) == 0:
        print('empty dataset'); return

    meta_path = args.data + '.meta.json'
    if not os.path.exists(meta_path):
        print('meta not found', meta_path); return
    import json
    meta = json.load(open(meta_path))
    vocab_size = meta['vocab_size']

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = LSTMModel(vocab_size=vocab_size, out_size=Y.size(1))
    device = torch.device('cpu')
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb,yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f'ep {ep+1}/{args.epochs} loss={total_loss/len(ds):.4f}')

    torch.save({'model_state': model.state_dict(), 'meta': meta}, args.out)
    print('saved', args.out)

if __name__ == '__main__':
    main()
