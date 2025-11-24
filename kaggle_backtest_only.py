"""
BACKTEST ONLY - KAGGLE
Usa el modelo ya entrenado para simular trading.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm.notebook import tqdm
import xgboost as xgb
import zipfile
import matplotlib.pyplot as plt

print("="*70)
print("  BACKTEST TREND FOLLOWING (MODELO PRE-ENTRENADO)")
print("="*70)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

# Rutas (AJUSTAR SI ES NECESARIO)
# Busca en /kaggle/input/ donde está tu modelo
MODEL_DIR = list(Path('/kaggle/input').glob('**/trend_model_triple_barrier.joblib'))
if not MODEL_DIR:
    raise FileNotFoundError("No se encontró el modelo. Asegúrate de agregar el Output del notebook anterior.")
MODEL_PATH = MODEL_DIR[0]

DATA_PATH = Path('/kaggle/input/trading-raw-data-compressed')

print(f"✓ Modelo encontrado: {MODEL_PATH}")
print(f"✓ Datos encontrados: {DATA_PATH}")

# ==========================================
# 2. CARGAR MODELO Y DATOS
# ==========================================

print("\n[Cargando recursos...]")
model = joblib.load(MODEL_PATH)

# Descomprimir datos si es necesario
zip_files = list(DATA_PATH.glob('*.zip'))
if zip_files:
    temp_path = Path('/kaggle/working/data')
    temp_path.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(temp_path)
    DATA_PATH = temp_path

csv_files = list(DATA_PATH.glob('*_history.csv'))
print(f"✓ {len(csv_files)} tickers disponibles")

# ==========================================
# 3. FUNCIONES DE PROCESAMIENTO
# ==========================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def load_ticker(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Features (IGUAL QUE EN ENTRENAMIENTO)
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

# ==========================================
# 4. MOTOR DE BACKTEST
# ==========================================

def run_backtest(start_date='2024-01-01'):
    INIT_CASH = 10000.0
    MAX_POSITIONS = 5
    MIN_CONFIDENCE = 0.60  # Ajustable
    K_ATR = 3.0
    HOLDING_MIN = 15
    
    cash = INIT_CASH
    positions = {}
    trades = []
    equity_curve = []
    
    # 1. Pre-procesar datos (vectorizado)
    print(f"\nGenerando señales para {len(csv_files)} tickers...")
    market_data = {}
    all_dates = set()
    
    # Detectar features usadas por el modelo
    model_features = model.get_booster().feature_names
    
    for f in tqdm(csv_files, desc="Prediciendo"):
        try:
            df = load_ticker(f)
            df = df[df['date'] >= start_date].copy()
            if df.empty: continue
            
            # Asegurar columnas
            for col in model_features:
                if col not in df.columns:
                    df[col] = 0 # O manejar error
            
            # Predecir
            probs = model.predict_proba(df[model_features])[:, 1]
            df['signal_prob'] = probs
            
            df = df.set_index('date').sort_index()
            market_data[f.stem.replace('_history', '')] = df
            all_dates.update(df.index)
        except Exception as e:
            continue
            
    sorted_dates = sorted(list(all_dates))
    print(f"Simulando {len(sorted_dates)} días...")
    
    # 2. Loop de simulación
    for current_date in sorted_dates:
        # --- GESTIÓN DE POSICIONES ---
        current_equity = cash + sum(p['shares'] * market_data[t].loc[current_date]['close'] 
                                  for t, p in positions.items() 
                                  if current_date in market_data[t].index)
        
        tickers_to_remove = []
        
        for ticker, pos in positions.items():
            if current_date not in market_data[ticker].index: continue
            
            row = market_data[ticker].loc[current_date]
            price = row['close']
            
            # Trailing Stop Update
            if price > pos['highest']:
                pos['highest'] = price
                new_stop = price - (K_ATR * row['atr'])
                pos['stop'] = max(pos['stop'], new_stop)
            
            pos['days'] += 1
            
            # Salidas
            exit = False
            reason = ""
            
            if price <= pos['stop']:
                exit = True
                reason = "Stop Loss"
            elif pos['days'] > HOLDING_MIN and price < row['ma_50']:
                exit = True
                reason = "Trend Broken"
                
            if exit:
                cash += pos['shares'] * price
                trades.append({
                    'ticker': ticker,
                    'entry': pos['entry_date'],
                    'exit': current_date,
                    'profit': (price - pos['entry_price']) * pos['shares'],
                    'return': (price - pos['entry_price']) / pos['entry_price'],
                    'reason': reason
                })
                tickers_to_remove.append(ticker)
                
        for t in tickers_to_remove: del positions[t]
        
        # --- NUEVAS ENTRADAS ---
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for ticker, df in market_data.items():
                if ticker in positions: continue
                if current_date not in df.index: continue
                
                row = df.loc[current_date]
                if row['signal_prob'] > MIN_CONFIDENCE and row['close'] > row['ma_200']:
                    candidates.append((ticker, row['signal_prob'], row))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(min(MAX_POSITIONS - len(positions), len(candidates))):
                ticker, prob, row = candidates[i]
                price = row['close']
                size = current_equity / MAX_POSITIONS
                shares = size / price
                
                cash -= shares * price
                positions[ticker] = {
                    'entry_date': current_date,
                    'entry_price': price,
                    'shares': shares,
                    'stop': price - (K_ATR * row['atr']),
                    'highest': price,
                    'days': 0
                }
        
        equity_curve.append({'date': current_date, 'equity': current_equity})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)

# ==========================================
# 5. EJECUTAR Y REPORTAR
# ==========================================

trades_df, equity_df = run_backtest('2024-01-01')

if not trades_df.empty:
    print("\n" + "="*40)
    print("  RESULTADOS 2024")
    print("="*40)
    final_eq = equity_df.iloc[-1]['equity']
    ret = (final_eq - 10000) / 10000
    
    print(f"Retorno Total: {ret:.2%}")
    print(f"Capital Final: ${final_eq:,.2f}")
    print(f"Trades:        {len(trades_df)}")
    print(f"Win Rate:      {(trades_df['profit']>0).mean():.2%}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(equity_df['date']), equity_df['equity'])
    plt.title(f"Equity Curve (Return: {ret:.1%})")
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No se generaron trades.")
