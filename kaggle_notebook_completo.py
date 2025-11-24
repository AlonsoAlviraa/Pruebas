"""
NOTEBOOK COMPLETO KAGGLE - TODO EN UNO
1. Genera dataset Triple Barrier
2. Entrena modelo con Optuna
    
    if pd.isna(entry_atr) or entry_atr <= 0:
        return 0, 0, 0.0
    
    upper_barrier = entry_price + (k_tp * entry_atr)
    lower_barrier = entry_price - (k_sl * entry_atr)
    
    for i in range(1, min(max_hold + 1, len(data) - entry_idx)):
        current_idx = entry_idx + i
        high = data.iloc[current_idx]['high']
        low = data.iloc[current_idx]['low']
        
        if high >= upper_barrier:
            return_pct = (upper_barrier - entry_price) / entry_price
            return 1, i, return_pct  # BUY
        
        if low <= lower_barrier:
            return_pct = (lower_barrier - entry_price) / entry_price
            return -1, i, return_pct  # SELL
    
    final_idx = min(entry_idx + max_hold, len(data) - 1)
    final_price = data.iloc[final_idx]['close']
    return_pct = (final_price - entry_price) / entry_price
    return 0, max_hold, return_pct  # HOLD


def load_and_process_ticker(ticker_file, data_path):
    """Carga un ticker y calcula features básicas"""
    df = pd.read_csv(ticker_file)
    
    # Asume que tienes OHLCV
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calcular ATR (necesario para barriers)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Calcular features básicas (puedes agregar más)
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    return df.dropna()


def compute_rsi(series, period=14):
    """Calcula RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def generate_dataset_from_csvs(data_path, k_tp=3.0, k_sl=2.0, max_hold=20):
    """Genera dataset completo desde archivos CSV"""
    
    # Buscar todos los CSV
    csv_files = list(data_path.glob('*_history.csv'))
    
    print(f"Encontrados {len(csv_files)} archivos CSV")
    
    all_labels = []
    
    for csv_file in tqdm(csv_files, desc="Procesando tickers"):
        ticker = csv_file.stem.replace('_history', '')
        
        try:
            df = load_and_process_ticker(csv_file, data_path)
            
            if df.empty or len(df) < 100:
                continue
            
            # Generar labels
            for idx in range(0, len(df) - max_hold - 1, 5):  # Cada 5 días
                label, holding_days, return_pct = calculate_triple_barrier_label(
                    df, idx, k_tp, k_sl, max_hold
                )
                
                if label != 0 or abs(return_pct) > 0.01:
                    all_labels.append({
                        'ticker': ticker,
                        'date': df.iloc[idx]['date'],
                        'label': label,
                        'holding_days': holding_days,
                        'return_pct': return_pct,
                        'close': df.iloc[idx]['close'],
                        'atr': df.iloc[idx]['atr'],
                        'ma_50': df.iloc[idx]['ma_50'],
                        'ma_200': df.iloc[idx]['ma_200'],
                        'rsi_14': df.iloc[idx]['rsi_14'],
                        'volume_sma': df.iloc[idx]['volume_sma'],
                    })
        
        except Exception as e:
            print(f"Error en {ticker}: {e}")
            continue
    
    dataset = pd.DataFrame(all_labels)
    
    print(f"\n✓ Dataset generado: {len(dataset):,} samples")
    print(f"  Tickers: {dataset['ticker'].nunique()}")
    
    # Estadísticas
    label_dist = dataset['label'].value_counts(normalize=True)
    print(f"\nDistribución:")
    for label, pct in label_dist.items():
        label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
        print(f"  {label_name}: {pct:.2%}")
    
    return dataset


# Generar dataset
dataset = generate_dataset_from_csvs(
    DATA_PATH,
    k_tp=3.0,
    k_sl=2.0,
    max_hold=20
)

# Guardar dataset intermedio
dataset.to_csv('/kaggle/working/triple_barrier_dataset.csv', index=False)
print("\n✓ Dataset guardado")

# ==========================================
# PARTE 2: ENTRENAR MODELO
# ==========================================

print("\n" + "="*70)
print("  PARTE 2: ENTRENANDO MODELO")
print("="*70)

# Preparar features
exclude_cols = ['date', 'ticker', 'label', 'holding_days', 'return_pct']
feature_cols = [c for c in dataset.columns if c not in exclude_cols]

X = dataset[feature_cols]
y = (dataset['label'] == 1).astype(int)  # BUY vs REST

print(f"\n✓ Features: {len(feature_cols)}")
print(f"  {feature_cols}")

# Split temporal
dataset = dataset.sort_values('date')
split_date = pd.Timestamp('2024-01-01')

train_mask = dataset['date'] < split_date
test_mask = dataset['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Optimizar con Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
    
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds))
    
    return np.mean(f1_scores)

print("\n[Optimizando hiperparámetros con Optuna...]")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n✓ Mejor F1: {study.best_value:.4f}")

# Entrenar modelo final
final_model = xgb.XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# Evaluar
y_pred_test = final_model.predict(X_test)
f1_test = f1_score(y_test, y_pred_test)

print("\n" + "="*70)
print("  RESULTADOS FINALES")
print("="*70)
print(classification_report(y_test, y_pred_test, target_names=['NO-BUY', 'BUY']))
print(f"\n✓ F1-Score Test: {f1_test:.4f}")

# ==========================================
# PARTE 3: GUARDAR MODELO
# ==========================================

# Guardar modelo
joblib.dump(final_model, '/kaggle/working/trend_model_triple_barrier.joblib')

# Guardar metadata
metadata = {
    'f1_score_test': float(f1_test),
    'best_params': study.best_params,
    'features': feature_cols,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'triple_barrier_params': {
        'k_tp': 3.0,
        'k_sl': 2.0,
        'max_hold': 20
    }
}

import json
with open('/kaggle/working/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*70)
print("  ✓ COMPLETADO")
print("="*70)
print("\nDescarga estos archivos desde /kaggle/working/:")
print("  - trend_model_triple_barrier.joblib")
print("  - model_metadata.json")
print("  - triple_barrier_dataset.csv")
