#date: 2026-01-02T17:01:37Z
#url: https://api.github.com/gists/126d7178ee8d480dc087633c9dbf6fb2
#owner: https://api.github.com/users/aldenpark

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from adapters import StocksAdapter, EveMarketHistoryAdapter

def build_features(df):
    df = df.sort_values(["asset_id", "ts"]).copy()
    df["ret_1"] = df.groupby("asset_id")["close"].pct_change()
    df["ret_5"] = df.groupby("asset_id")["close"].pct_change(5)
    df["vol_5"] = df.groupby("asset_id")["close"].rolling(5).std().reset_index(0, drop=True)
    df = df.dropna()
    return df

def make_windows(df, lookback=20, horizons=(1, 7)):
    X, y = [], []
    for _, g in df.groupby("asset_id"):
        g = g.sort_values("ts")
        feats = g[["open","high","low","close","volume","ret_1","ret_5","vol_5"]].to_numpy()
        close = g["close"].to_numpy()
        for i in range(len(feats) - lookback - max(horizons)):
            X.append(feats[i:i+lookback])
            targets = []
            for h in horizons:
                ret = (close[i+lookback+h-1] / close[i+lookback-1]) - 1.0
                targets.append(ret)
            y.append(targets)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

def train(df):
    df = build_features(df)
    X, y = make_windows(df, lookback=20, horizons=(1, 7))

    split = int(len(X) * 0.8)  # walk‑forward by time in practice
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = LSTMRegressor(n_features=X.shape[-1], out_dim=2)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)
        print(f"epoch={epoch+1} val_mse={val_loss:.4f}")

if __name__ == "__main__":
    # df = StocksAdapter("stocks.csv").load()
    df = EveMarketHistoryAdapter("eve_market_history.csv").load()
    train(df)