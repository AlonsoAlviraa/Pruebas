"""
KAGGLE GRID SEARCH BACKTESTER
Prueba múltiples combinaciones de estrategias usando tu modelo entrenado.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm.notebook import tqdm
import zipfile
import itertools
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("  GRID SEARCH STRATEGY OPTIMIZER")
print("="*70)

# ==========================================
# 1. CONFIGURACIÓN Y CARGA
# ==========================================

# Buscar Modelo
MODEL_FILES = list(Path('/kaggle/input').glob('**/*.joblib'))
if not MODEL_FILES:
    raise FileNotFoundError("❌ No se encontró el modelo .joblib en Input")
MODEL_PATH = MODEL_FILES[0]
print(f"✓ Modelo: {MODEL_PATH.name}")

# Buscar Datos
DATA_PATH = Path('/kaggle/input/trading-raw-data-compressed')
zip_files = list(DATA_PATH.glob('*.zip'))

if zip_files:
    print(f"✓ Descomprimiendo datos...")
    temp_path = Path('/kaggle/working/data')
    temp_path.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(temp_path)
    DATA_PATH = temp_path

csv_files = list(DATA_PATH.glob('*_history.csv'))
print(f"✓ Tickers: {len(csv_files)}")

# Cargar Modelo
model = joblib.load(MODEL_PATH)

# ==========================================
# 2. PRE-PROCESAMIENTO MASIVO
# ==========================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_ticker(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Features (IGUAL QUE TRAINING)
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df.dropna()

print("\n[Pre-calculando señales para todos los tickers...]")
market_data = {}
model_features = model.get_booster().feature_names

# Cargar y predecir SOLO UNA VEZ
for f in tqdm(csv_files, desc="Cargando"):
    try:
        df = process_ticker(f)
        # Filtrar solo 2024 para backtest rápido
        df = df[df['date'] >= '2024-01-01'].copy()
        if df.empty: continue
        
        # Asegurar columnas
        for col in model_features:
            if col not in df.columns: df[col] = 0
            
        # Predecir
        df['signal_prob'] = model.predict_proba(df[model_features])[:, 1]
        
        df = df.set_index('date').sort_index()
        market_data[f.stem.replace('_history', '')] = df
    except:
        continue

print(f"✓ Datos listos: {len(market_data)} tickers en memoria")

# ==========================================
# 3. MOTOR DE BACKTEST (RÁPIDO)
# ==========================================

def run_fast_backtest(params):
    # Desempaquetar parámetros
    MIN_CONF = params['min_conf']
    K_ATR = params['k_atr']
    HOLD_MIN = params['hold_min']
    MAX_POS = params['max_pos']
    
    cash = 10000.0
    positions = {}
    trades_count = 0
    wins = 0
    
    # Obtener todas las fechas únicas ordenadas
    all_dates = sorted(list(set(d for df in market_data.values() for d in df.index)))
    
    for current_date in all_dates:
        # 1. GESTIÓN POSICIONES
        current_equity = cash + sum(p['shares'] * market_data[t].loc[current_date]['close'] 
                                  for t, p in positions.items() 
                                  if current_date in market_data[t].index)
        
        remove_tickers = []
        for ticker, pos in positions.items():
            if current_date not in market_data[ticker].index: continue
            row = market_data[ticker].loc[current_date]
            price = row['close']
            
            # Trailing Stop
            if price > pos['highest']:
                pos['highest'] = price
                pos['stop'] = max(pos['stop'], price - (K_ATR * row['atr']))
            
            pos['days'] += 1
            
            # Salida
            if price <= pos['stop'] or (pos['days'] > HOLD_MIN and price < row['ma_50']):
                cash += pos['shares'] * price
                trades_count += 1
                if price > pos['entry']: wins += 1
                remove_tickers.append(ticker)
        
        for t in remove_tickers: del positions[t]
        
        # 2. ENTRADAS
        if len(positions) < MAX_POS:
            candidates = []
            for ticker, df in market_data.items():
                if ticker in positions: continue
                if current_date not in df.index: continue
                row = df.loc[current_date]
                
                if row['signal_prob'] >= MIN_CONF and row['close'] > row['ma_200']:
                    candidates.append((ticker, row['signal_prob'], row))
            
            # Ordenar por probabilidad
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(min(MAX_POS - len(positions), len(candidates))):
                ticker, prob, row = candidates[i]
                price = row['close']
                size = current_equity / MAX_POS
                shares = size / price
                cash -= shares * price
                positions[ticker] = {
                    'entry': price, 'shares': shares, 
                    'stop': price - (K_ATR * row['atr']),
                    'highest': price, 'days': 0
                }
                
    final_equity = cash + sum(p['shares'] * market_data[t].iloc[-1]['close'] for t, p in positions.items())
    return {
        'Return': (final_equity - 10000) / 10000,
        'Trades': trades_count,
        'WinRate': wins / trades_count if trades_count > 0 else 0,
        'FinalEquity': final_equity
    }

# ==========================================
# 4. GRID SEARCH
# ==========================================

# DEFINIR GRID DE PARÁMETROS
param_grid = {
    'min_conf': [0.55, 0.60, 0.65],
    'k_atr': [2.0, 2.5, 3.0],
    'hold_min': [10, 20],
    'max_pos': [5, 10]
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\n[Probando {len(combinations)} combinaciones...]")

results = []
for params in tqdm(combinations):
    res = run_fast_backtest(params)
    results.append({**params, **res})

# ==========================================
# 5. RESULTADOS
# ==========================================

df_res = pd.DataFrame(results)
df_res = df_res.sort_values('Return', ascending=False)

print("\n" + "="*70)
print("  TOP 10 CONFIGURACIONES")
print("="*70)
print(df_res.head(10).to_string(index=False))

# Guardar
df_res.to_csv('grid_search_results.csv', index=False)
print("\n✓ Resultados guardados en grid_search_results.csv")

# Plot mejor resultado
best = df_res.iloc[0]
print(f"\n🏆 MEJOR CONFIGURACIÓN:")
print(best)
