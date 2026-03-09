import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

def parse_date(d_str):
    p = str(d_str).split()
    if len(p) >= 4:
        m_map = {'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04', 'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08', 'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'}
        return f"{p[3]}-{m_map.get(p[2].lower(), '01')}-{p[1].zfill(2)}"
    return None

def load_data(p):
    with open(p, 'r', encoding='latin1') as f:
        idx = next((i for i, ln in enumerate(f) if 'secteur' in ln.lower()), 0)
    df = pd.read_csv(p, sep=';', encoding='latin1', skiprows=idx)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['tstamp'].apply(parse_date), errors='coerce')
    df['trafic_mbps'] = pd.to_numeric(df['trafic_mbps'], errors='coerce')
    return df.dropna(subset=['date', 'trafic_mbps', 'secteur']).sort_values(['secteur', 'date'])

def augment(series, splits=10, var_ratio=0.05):
    aug = []
    for val in series:
        sim = np.random.normal(loc=val, scale=val * var_ratio, size=splits - 1)
        sim = np.maximum(sim, val * 0.01)
        aug.extend([val] + sim.tolist())
    return np.array(aug)

def run_sarima(train, steps):
    mod = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    res = mod.fit(disp=False)
    return res.forecast(steps)

def main():
    # Use relative path or standard project path
    data_path = os.path.join("project", "histo_trafic.csv")
    if not os.path.exists(data_path):
        # Fallback to absolute path if running from elsewhere, but usually we run from project root
        data_path = "c:/Users/PC/Desktop/chronos/project/histo_trafic.csv"
        
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows.")

    print("Loading Chronos model...")
    # Use 'auto' for device map if available, else cpu. 
    # Small model usually fits on CPU or GPU easily.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=device, torch_dtype=torch.float32)

    c_preds, s_preds, truths = [], [], []

    sectors = df['secteur'].unique()
    print(f"Processing {len(sectors)} sectors...")

    for i, (sec, grp) in enumerate(df.groupby("secteur")):
        if i % 10 == 0:
            print(f"Processing sector {i}/{len(sectors)}: {sec}")
            
        grp = grp.drop_duplicates('date').set_index('date').asfreq('7D')
        if len(grp) < 5:
            continue
        
        grp['trafic_mbps'] = grp['trafic_mbps'].interpolate().ffill().bfill()
        raw = grp['trafic_mbps'].values
        
        train_raw, test_raw = raw[:-1], raw[-1]
        train_aug = augment(train_raw)
        test_aug = augment([test_raw])
        
        ctx = torch.tensor(train_aug, dtype=torch.float32).unsqueeze(0)
        fcst = pipe.predict(ctx, prediction_length=10, num_samples=20)
        c_pred = np.median(fcst[0].numpy(), axis=0)
        
        s_pred = run_sarima(train_aug, 10)
        
        c_preds.extend(c_pred)
        s_preds.extend(s_pred)
        truths.extend(test_aug)

    if not truths:
        print("No valid data to evaluate.")
        return

    chronos_rmse = np.sqrt(mean_squared_error(truths, c_preds))
    sarima_rmse = np.sqrt(mean_squared_error(truths, s_preds))

    print(f"\n--- Final Results ---")
    print(f"Chronos RMSE: {chronos_rmse:.4f}")
    print(f"SARIMA RMSE: {sarima_rmse:.4f}")

if __name__ == "__main__":
    main()
