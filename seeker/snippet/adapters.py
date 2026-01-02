#date: 2026-01-02T17:01:37Z
#url: https://api.github.com/gists/126d7178ee8d480dc087633c9dbf6fb2
#owner: https://api.github.com/users/aldenpark

import pandas as pd

class BaseAdapter:
    def load(self) -> pd.DataFrame:
        """Return DataFrame with columns: ts, open, high, low, close, volume, asset_id."""
        raise NotImplementedError

class StocksAdapter(BaseAdapter):
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        df = pd.read_csv(self.csv_path)
        # Expected columns: date, open, high, low, close, volume, symbol
        df = df.rename(columns={"date": "ts", "symbol": "asset_id"})
        return df[["ts", "open", "high", "low", "close", "volume", "asset_id"]]

class EveMarketHistoryAdapter(BaseAdapter):
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        df = pd.read_csv(self.csv_path)
        # market_history schema: region_id, type_id, date, average, highest, lowest, volume, order_count
        df = df.rename(columns={
            "date": "ts",
            "average": "close",
            "highest": "high",
            "lowest": "low"
        })
        # EVE table does not store open; approximate with previous close per type_id/region_id
        df = df.sort_values(["region_id", "type_id", "ts"])
        df["open"] = df.groupby(["region_id", "type_id"])["close"].shift(1)
        df["open"] = df["open"].fillna(df["close"])

        # asset_id combines region + type to keep series distinct
        df["asset_id"] = df["region_id"].astype(str) + ":" + df["type_id"].astype(str)

        return df[["ts", "open", "high", "low", "close", "volume", "asset_id"]]
