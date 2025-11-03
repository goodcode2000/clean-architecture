 import os
 import time
 import json
 import math
 import io
 from datetime import datetime, timedelta, timezone
 from typing import List, Dict, Tuple, Optional
 
 import numpy as np
 import pandas as pd
 import requests
 from dateutil import tz
 
 import pandas_ta as ta
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_absolute_error
 from sklearn.preprocessing import StandardScaler
 import lightgbm as lgb
 
 import statsmodels.api as sm
 from statsmodels.tsa.holtwinters import ExponentialSmoothing
 from arch import arch_model
 
 import torch
 import torch.nn as nn
 
 DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
 RAW_DATA_CSV = os.path.join(DATA_DIR, "btc_5m.csv")
 PRED_LOG_CSV = os.path.join(DATA_DIR, "predictions.csv")
 
 SYMBOL = "BTCUSDT"
 INTERVAL = "5m"
 WINDOW_M = 5
 RESAMPLE_SECONDS = 300
 HISTORY_DAYS = 90
 
 os.makedirs(DATA_DIR, exist_ok=True)
 
 def utcnow() -> datetime:
     return datetime.now(timezone.utc)
 
 def floor_to_5m(ts: datetime) -> datetime:
     seconds = int(ts.timestamp())
     floored = seconds - (seconds % RESAMPLE_SECONDS)
     return datetime.fromtimestamp(floored, tz=timezone.utc)
 
 def fetch_binance_klines(symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> List[List]:
     url = "https://api.binance.com/api/v3/klines"
     params = {
         "symbol": symbol,
         "interval": interval,
         "startTime": start_time_ms,
         "endTime": end_time_ms,
         "limit": 1000,
     }
     resp = requests.get(url, params=params, timeout=30)
     resp.raise_for_status()
     return resp.json()
 
 def load_or_fetch_history() -> pd.DataFrame:
     if os.path.exists(RAW_DATA_CSV):
         df = pd.read_csv(RAW_DATA_CSV, parse_dates=["open_time"])
         df.sort_values("open_time", inplace=True)
     else:
         df = pd.DataFrame()
     
     now_utc = utcnow()
     end = floor_to_5m(now_utc)
     start = end - timedelta(days=HISTORY_DAYS)
     
     if df.empty or df["open_time"].max(tz=None) < start:
         # Fresh fetch for the entire 90 days window
         all_rows: List[List] = []
         cur = int(start.timestamp() * 1000)
         end_ms = int(end.timestamp() * 1000)
         while cur < end_ms:
             chunk = fetch_binance_klines(SYMBOL, "5m", cur, end_ms)
             if not chunk:
                 break
             all_rows.extend(chunk)
             last_open = chunk[-1][0]
             cur = last_open + RESAMPLE_SECONDS * 1000
             time.sleep(0.1)
         df = klines_to_df(all_rows)
         df.to_csv(RAW_DATA_CSV, index=False)
     else:
         # Incremental update from last timestamp
         last_ts = pd.to_datetime(df["open_time"].max()).tz_localize(timezone.utc)
         start_ms = int((last_ts + timedelta(seconds=RESAMPLE_SECONDS)).timestamp() * 1000)
         end_ms = int(end.timestamp() * 1000)
         if start_ms <= end_ms:
             new_rows: List[List] = []
             cur = start_ms
             while cur <= end_ms:
                 chunk = fetch_binance_klines(SYMBOL, "5m", cur, end_ms)
                 if not chunk:
                     break
                 new_rows.extend(chunk)
                 cur = chunk[-1][0] + RESAMPLE_SECONDS * 1000
                 time.sleep(0.1)
             if new_rows:
                 new_df = klines_to_df(new_rows)
                 df = pd.concat([df, new_df], ignore_index=True)
                 df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
                 df.sort_values("open_time", inplace=True)
                 # keep last 90 days only
                 df = df[df["open_time"] >= (end - timedelta(days=HISTORY_DAYS))]
                 df.to_csv(RAW_DATA_CSV, index=False)
     return df
 
 def klines_to_df(klines: List[List]) -> pd.DataFrame:
     cols = [
         "open_time","open","high","low","close","volume",
         "close_time","quote_asset_volume","number_of_trades",
         "taker_buy_base","taker_buy_quote","ignore"
     ]
     df = pd.DataFrame(klines, columns=cols)
     for c in ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
         df[c] = df[c].astype(float)
     df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
     df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
     df["number_of_trades"] = df["number_of_trades"].astype(int)
     df = df[["open_time","open","high","low","close","volume","number_of_trades","quote_asset_volume","taker_buy_base","taker_buy_quote"]]
     return df
 
 def resample_and_features(df: pd.DataFrame) -> pd.DataFrame:
     df = df.copy()
     df.set_index("open_time", inplace=True)
     df = df.resample(f"{RESAMPLE_SECONDS}s").agg({
         "open":"first","high":"max","low":"min","close":"last","volume":"sum",
         "number_of_trades":"sum","quote_asset_volume":"sum","taker_buy_base":"sum","taker_buy_quote":"sum"
     }).dropna()
     df["return"] = df["close"].pct_change()
     df["log_return"] = np.log(df["close"]).diff()
     # Technical indicators
     df["rsi_14"] = ta.rsi(df["close"], length=14)
     bb = ta.bbands(df["close"], length=20, std=2.0)
     if bb is not None:
         df["bb_low"], df["bb_mid"], df["bb_high"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
     macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
     if macd is not None:
         df["macd"], df["macd_signal"], df["macd_hist"] = macd.iloc[:,0], macd.iloc[:,1], macd.iloc[:,2]
     df["volatility_5"] = df["log_return"].rolling(5).std() * np.sqrt(RESAMPLE_SECONDS)
     df["volatility_12"] = df["log_return"].rolling(12).std() * np.sqrt(RESAMPLE_SECONDS)
     df["rv_5"] = df["return"].rolling(5).apply(lambda x: np.sqrt((x**2).sum()), raw=True)
     # Liquidity proxies
     df["dollar_vol"] = df["quote_asset_volume"]
     df["trade_intensity"] = df["number_of_trades"] / RESAMPLE_SECONDS
     # Shift target: next 5-minute close
     df["target_close"] = df["close"].shift(-1)
     df.dropna(inplace=True)
     return df
 
 class ETSForecaster:
     def __init__(self):
         self.model = None
         self.fitted = None
     
     def fit(self, series: pd.Series):
         self.model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
         self.fitted = self.model.fit(optimized=True)
     
     def predict_next(self) -> float:
         if self.fitted is None:
             raise RuntimeError("ETS not fitted")
         return float(self.fitted.forecast(1).iloc[0])
 
 class GARCHForecaster:
     def __init__(self):
         self.res = None
         self.last_price = None
     
     def fit(self, log_returns: pd.Series, last_price: float):
         lr = log_returns.dropna() * 100.0
         if len(lr) < 50:
             self.res = None
             self.last_price = last_price
             return
         am = arch_model(lr.values, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
         self.res = am.fit(disp='off')
         self.last_price = last_price
     
     def predict_next(self) -> float:
         if self.res is None or self.last_price is None:
             return float(self.last_price) if self.last_price is not None else np.nan
         # zero-mean return, use variance shock to adjust slightly
         fvar = self.res.forecast(horizon=1).variance.values[-1, 0] / (100.0**2)
         adj = 0.0
         # heuristic: if variance high, expect mean-reverting small move
         adj = -0.05 * np.sqrt(max(fvar, 1e-12)) * self.last_price
         return float(self.last_price + adj)
 
 class LGBMRegressor:
     def __init__(self):
         self.model = None
         self.scaler = None
         self.feature_names: List[str] = []
     
     def fit(self, df: pd.DataFrame):
         feature_cols = [c for c in df.columns if c not in ["target_close"]]
         X = df[feature_cols]
         y = df["target_close"]
         self.scaler = StandardScaler()
         Xs = self.scaler.fit_transform(X)
         X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, shuffle=False)
         lgb_train = lgb.Dataset(X_train, label=y_train)
         lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
         params = {
             "objective": "regression",
             "metric": "mae",
             "learning_rate": 0.05,
             "num_leaves": 31,
             "feature_fraction": 0.9,
             "bagging_fraction": 0.8,
             "bagging_freq": 1,
             "verbose": -1,
         }
         self.model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=600, early_stopping_rounds=50, verbose_eval=False)
         self.feature_names = feature_cols
     
     def predict_row(self, row: pd.Series) -> float:
         X = row[self.feature_names].values.reshape(1, -1)
         Xs = self.scaler.transform(X)
         return float(self.model.predict(Xs)[0])
 
 class SmallLSTM(nn.Module):
     def __init__(self, input_dim: int, hidden: int = 32):
         super().__init__()
         self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
         self.fc = nn.Linear(hidden, 1)
     def forward(self, x):
         out, _ = self.lstm(x)
         return self.fc(out[:, -1, :])
 
 class SmallCNN1D(nn.Module):
     def __init__(self, input_dim: int):
         super().__init__()
         self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
         self.relu = nn.ReLU()
         self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
         self.pool = nn.AdaptiveAvgPool1d(1)
         self.fc = nn.Linear(8, 1)
     def forward(self, x):
         # x: (B, T, F) -> (B, F, T)
         x = x.permute(0, 2, 1)
         x = self.relu(self.conv1(x))
         x = self.relu(self.conv2(x))
         x = self.pool(x).squeeze(-1)
         return self.fc(x)
 
 class SeqModels:
     def __init__(self, seq_len: int, feature_cols: List[str]):
         self.seq_len = seq_len
         self.feature_cols = feature_cols
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.lstm = SmallLSTM(len(feature_cols)).to(self.device)
         self.cnn = SmallCNN1D(len(feature_cols)).to(self.device)
         self.scaler = StandardScaler()
     
     def _make_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
         X = df[self.feature_cols].values
         y = df["target_close"].values
         Xs = self.scaler.fit_transform(X)
         seq_X, seq_y = [], []
         for i in range(self.seq_len, len(Xs)):
             seq_X.append(Xs[i-self.seq_len:i])
             seq_y.append(y[i])
         return np.array(seq_X, dtype=np.float32), np.array(seq_y, dtype=np.float32)
     
     def fit(self, df: pd.DataFrame, epochs: int = 5):
         if len(df) < self.seq_len + 10:
             return
         X, y = self._make_sequences(df)
         X_t = torch.from_numpy(X).to(self.device)
         y_t = torch.from_numpy(y).unsqueeze(1).to(self.device)
         crit = nn.L1Loss()
         # Train LSTM
         opt = torch.optim.Adam(self.lstm.parameters(), lr=1e-3)
         self.lstm.train()
         for _ in range(epochs):
             opt.zero_grad()
             pred = self.lstm(X_t)
             loss = crit(pred, y_t)
             loss.backward()
             opt.step()
         # Train CNN
         opt2 = torch.optim.Adam(self.cnn.parameters(), lr=1e-3)
         self.cnn.train()
         for _ in range(epochs):
             opt2.zero_grad()
             pred = self.cnn(X_t)
             loss = crit(pred, y_t)
             loss.backward()
             opt2.step()
     
     def predict_row(self, df: pd.DataFrame) -> Tuple[float, float]:
         # Use last seq_len rows
         if len(df) < self.seq_len + 1:
             last_price = float(df["close"].iloc[-1])
             return last_price, last_price
         X = df[self.feature_cols].values
         Xs = self.scaler.transform(X)
         seq = Xs[-self.seq_len:]
         X_t = torch.from_numpy(seq[np.newaxis, ...].astype(np.float32)).to(self.device)
         self.lstm.eval(); self.cnn.eval()
         with torch.no_grad():
             p_lstm = float(self.lstm(X_t).cpu().numpy().ravel()[0])
             p_cnn = float(self.cnn(X_t).cpu().numpy().ravel()[0])
         return p_lstm, p_cnn
 
 class OnlineEnsemble:
     def __init__(self):
         # weights for [ETS, GARCH, LGBM, LSTM, CNN, Residual]
         self.weights = np.array([0.15, 0.10, 0.35, 0.2, 0.2, 0.0], dtype=float)
         self.alpha = 0.2  # learning rate for weight update
         self.eps = 1e-6
         self.residual_lgbm = None
         self.residual_features: List[str] = []
     
     def set_residual_model(self, model, feature_names: List[str]):
         self.residual_lgbm = model
         self.residual_features = feature_names
     
     def combine(self, components: Dict[str, float], row: Optional[pd.Series] = None) -> float:
         preds = np.array([
             components.get("ets", np.nan),
             components.get("garch", np.nan),
             components.get("lgbm", np.nan),
             components.get("lstm", np.nan),
             components.get("cnn", np.nan),
             0.0
         ], dtype=float)
         # residual correction
         if self.residual_lgbm is not None and row is not None and set(self.residual_features).issubset(row.index):
             res_in = row[self.residual_features].values.reshape(1, -1)
             try:
                 res_adj = float(self.residual_lgbm.predict(res_in)[0])
             except Exception:
                 res_adj = 0.0
             preds[-1] = res_adj
         mask = ~np.isnan(preds)
         w = self.weights.copy()
         w[~mask] = 0.0
         if w.sum() == 0:
             return float(np.nanmean(preds))
         w = w / (w.sum() + self.eps)
         return float(np.nansum(preds * w))
     
     def update_weights(self, components_history: List[Dict[str, float]], y_true: List[float]):
         # error-weighted update over recent window
         if len(components_history) == 0:
             return
         comp_keys = ["ets","garch","lgbm","lstm","cnn","residual"]
         errs = np.zeros(len(comp_keys), dtype=float)
         cnts = np.zeros(len(comp_keys), dtype=float)
         for comp, yt in zip(components_history, y_true):
             for i,k in enumerate(comp_keys):
                 if k in comp and not np.isnan(comp[k]):
                     errs[i] += abs(comp[k] - yt)
                     cnts[i] += 1
         avg_err = np.divide(errs, np.maximum(cnts, 1.0))
         inv_err = 1.0 / (avg_err + self.eps)
         new_w = inv_err / (inv_err.sum() + self.eps)
         self.weights = (1 - self.alpha) * self.weights + self.alpha * new_w
 
 def train_residual_correction(df: pd.DataFrame) -> Tuple[Optional[object], List[str]]:
     # Simple last-step features for residual correction
     feat_cols = [
         "rsi_14","macd","macd_signal","volatility_5","volatility_12","rv_5",
         "trade_intensity","dollar_vol","return"
     ]
     df2 = df.copy()
     df2["residual"] = df2["target_close"] - df2["close"]
     df2 = df2.dropna()
     if len(df2) < 200:
         return None, []
     X = df2[feat_cols].fillna(0.0).values
     y = df2["residual"].values
     model = lgb.LGBMRegressor(
         n_estimators=200,
         learning_rate=0.05,
         max_depth=-1,
         subsample=0.8,
         colsample_bytree=0.9,
         random_state=42
     )
     model.fit(X, y)
     return model, feat_cols
 
 def append_prediction_log(ts: datetime, real_price: Optional[float], pred_price: Optional[float]):
     exists = os.path.exists(PRED_LOG_CSV)
     with open(PRED_LOG_CSV, "a", encoding="utf-8") as f:
         if not exists:
             f.write("timestamp,real_price,pred_price,diff,line_index\n")
         line_count = sum(1 for _ in open(PRED_LOG_CSV, "r", encoding="utf-8")) if exists else 1
         rp = "" if real_price is None else f"{real_price:.2f}"
         pp = "" if pred_price is None else f"{pred_price:.2f}"
         diff = "" if (real_price is None or pred_price is None) else f"{(real_price - pred_price):.2f}"
         f.write(f"{ts.isoformat()},{rp},{pp},{diff},{line_count}\n")
 
 def main_loop():
     print("[BOOT] Loading 90-day 5m BTCUSDT data...")
     df_raw = load_or_fetch_history()
     df_feat = resample_and_features(df_raw)
     last_price = float(df_feat["close"].iloc[-1])
     last_ts = df_feat.index[-1]
     # At start, output only real price
     print(f"[REAL-ONLY] {last_ts.isoformat()} real_price={last_price:.2f}")
     append_prediction_log(last_ts, last_price, None)
     
     # Fit base models
     ets = ETSForecaster()
     ets.fit(df_feat["close"])  # level/trend smoothing
     garch = GARCHForecaster()
     garch.fit(df_feat["log_return"], last_price=last_price)
     
     # LGBM
     df_ml = df_feat.copy()
     target = df_ml["target_close"].copy()
     feature_cols = [c for c in df_ml.columns if c != "target_close"]
     lgbm_model = LGBMRegressor()
     lgbm_model.fit(df_ml[[*feature_cols, "target_close"]])
     
     # Sequence models
     seq_len = 24
     seq_features = [c for c in feature_cols if c not in ["target_close"]]
     seq_models = SeqModels(seq_len=seq_len, feature_cols=seq_features)
     seq_models.fit(df_ml[[*seq_features, "target_close"]], epochs=3)
     
     # Residual correction model
     res_model, res_feats = train_residual_correction(df_feat)
     ensemble = OnlineEnsemble()
     if res_model is not None:
         ensemble.set_residual_model(res_model, res_feats)
     
     components_hist: List[Dict[str, float]] = []
     y_hist: List[float] = []
     
     while True:
         try:
             # Sync to next 5-minute boundary
             now = utcnow()
             next_slot = floor_to_5m(now) + timedelta(seconds=RESAMPLE_SECONDS)
             sleep_s = (next_slot - now).total_seconds()
             if sleep_s > 0:
                 time.sleep(min(sleep_s, 300))
             
             # Update data
             df_raw = load_or_fetch_history()
             df_feat = resample_and_features(df_raw)
             last_price = float(df_feat["close"].iloc[-1])
             last_ts = df_feat.index[-1]
             
             # Predict next 5m close
             ets_pred = ets.predict_next()
             garch.fit(df_feat["log_return"], last_price=last_price)
             garch_pred = garch.predict_next()
             lgbm_pred = lgbm_model.predict_row(df_feat.iloc[-1])
             lstm_pred, cnn_pred = seq_models.predict_row(df_feat[[*seq_features, "target_close"]])
             components = {
                 "ets": ets_pred,
                 "garch": garch_pred,
                 "lgbm": lgbm_pred,
                 "lstm": lstm_pred,
                 "cnn": cnn_pred,
             }
             final_pred = ensemble.combine(components, df_feat.iloc[-1])
             
             # Output terminal comparison
             print(f"[PREDICT] {last_ts.isoformat()} real={last_price:.2f}  pred+5m={final_pred:.2f}  (ets={ets_pred:.2f}, garch={garch_pred:.2f}, lgbm={lgbm_pred:.2f}, lstm={lstm_pred:.2f}, cnn={cnn_pred:.2f})")
             append_prediction_log(last_ts, last_price, final_pred)
             
             # Update weights with known realized values if we have prior prediction
             if len(y_hist) >= 1:
                 ensemble.update_weights(components_hist[-len(y_hist):], y_hist)
             components_hist.append({**components, "residual": 0.0})
             y_hist.append(last_price)
             if len(y_hist) > 48:
                 y_hist = y_hist[-48:]
                 components_hist = components_hist[-48:]
         except KeyboardInterrupt:
             print("Exiting loop.")
             break
         except Exception as e:
             print(f"[ERROR] {e}")
             time.sleep(5)
 
 if __name__ == "__main__":
     main_loop()

