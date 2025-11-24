
# ==============================================
# PASO 6: BACKTEST EN KAGGLE (Trend Following)
# ==============================================

print("\n" + "="*70)
print("  PASO 6: EJECUTANDO BACKTEST (2024)")
print("="*70)

def run_kaggle_backtest(data_path, model, start_date='2024-01-01'):
    # Configuración Estrategia
    INIT_CASH = 10000.0
    MAX_POSITIONS = 5
    MIN_CONFIDENCE = 0.65
    K_ATR = 3.0  # Trailing stop ancho
    HOLDING_MIN = 20 # Días
    
    cash = INIT_CASH
    positions = {} # ticker -> {entry_price, shares, stop_loss, days_held, highest_price}
    trades = []
    equity_curve = []
    
    # Buscar CSVs
    csv_files = list(data_path.glob('*_history.csv'))
    print(f"Backtesting en {len(csv_files)} tickers desde {start_date}...")
    
    # Cargar todos los datos en memoria (optimización)
    market_data = {}
    all_dates = set()
    
    for f in tqdm(csv_files, desc="Cargando datos"):
        try:
            df = load_and_process_ticker(f)
            if df.empty: continue
            
            # Filtrar fecha
            df = df[df['date'] >= start_date].copy()
            if df.empty: continue
            
            # Predecir señal con el modelo
            features = [c for c in df.columns if c in feature_cols]
            if not features: continue
            
            # Predicción vectorizada
            probs = model.predict_proba(df[features])[:, 1]
            df['signal_prob'] = probs
            
            df = df.set_index('date').sort_index()
            market_data[f.stem.replace('_history', '')] = df
            all_dates.update(df.index)
        except:
            continue
            
    sorted_dates = sorted(list(all_dates))
    print(f"Simulando {len(sorted_dates)} días de trading...")
    
    # Bucle día a día
    for current_date in sorted_dates:
        # 1. Actualizar posiciones y chequear salidas
        current_equity = cash
        tickers_to_remove = []
        
        for ticker, pos in positions.items():
            if current_date not in market_data[ticker].index:
                continue
                
            row = market_data[ticker].loc[current_date]
            current_price = row['close']
            current_atr = row['atr']
            
            # Actualizar trailing stop
            if current_price > pos['highest_price']:
                pos['highest_price'] = current_price
                new_stop = current_price - (K_ATR * current_atr)
                pos['stop_loss'] = max(pos['stop_loss'], new_stop)
            
            pos['days_held'] += 1
            current_val = pos['shares'] * current_price
            current_equity += current_val
            
            # Chequear Salida
            exit_signal = False
            reason = ""
            
            # A) Trailing Stop
            if current_price <= pos['stop_loss']:
                exit_signal = True
                reason = "Stop Loss"
            
            # B) Tendencia Rota (Close < MA50) tras holding min
            elif pos['days_held'] > HOLDING_MIN and current_price < row['ma_50']:
                exit_signal = True
                reason = "Trend Broken"
                
            if exit_signal:
                # Vender
                cash += current_val
                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'profit': current_val - (pos['shares'] * pos['entry_price']),
                    'return': (current_price - pos['entry_price']) / pos['entry_price'],
                    'days': pos['days_held'],
                    'reason': reason
                })
                tickers_to_remove.append(ticker)
        
        for t in tickers_to_remove:
            del positions[t]
            
        # 2. Chequear Entradas
        if len(positions) < MAX_POSITIONS:
            # Buscar candidatos
            candidates = []
            for ticker, df in market_data.items():
                if ticker in positions: continue
                if current_date not in df.index: continue
                
                row = df.loc[current_date]
                
                # Filtros de entrada
                if (row['signal_prob'] >= MIN_CONFIDENCE and 
                    row['close'] > row['ma_200']): # Filtro de tendencia largo plazo
                    candidates.append((ticker, row['signal_prob'], row))
            
            # Ordenar por confianza
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Comprar top candidatos
            slots_available = MAX_POSITIONS - len(positions)
            for i in range(min(slots_available, len(candidates))):
                ticker, prob, row = candidates[i]
                price = row['close']
                atr = row['atr']
                
                # Size: Equal weight
                position_size = current_equity / MAX_POSITIONS
                shares = position_size / price
                
                if shares > 0:
                    cash -= (shares * price)
                    positions[ticker] = {
                        'entry_date': current_date,
                        'entry_price': price,
                        'shares': shares,
                        'stop_loss': price - (K_ATR * atr),
                        'highest_price': price,
                        'days_held': 0
                    }
        
        equity_curve.append({'date': current_date, 'equity': current_equity})

    # Reporte Final
    final_equity = equity_curve[-1]['equity'] if equity_curve else INIT_CASH
    total_return = (final_equity - INIT_CASH) / INIT_CASH
    
    print("\n" + "="*40)
    print("  RESULTADOS BACKTEST (2024)")
    print("="*40)
    print(f"Capital Inicial: ${INIT_CASH:,.2f}")
    print(f"Capital Final:   ${final_equity:,.2f}")
    print(f"Retorno Total:   {total_return:.2%}")
    print(f"Total Trades:    {len(trades)}")
    
    if trades:
        df_trades = pd.DataFrame(trades)
        win_rate = (df_trades['profit'] > 0).mean()
        avg_win = df_trades[df_trades['profit'] > 0]['profit'].mean()
        avg_loss = df_trades[df_trades['profit'] <= 0]['profit'].mean()
        
        print(f"Win Rate:        {win_rate:.2%}")
        print(f"Avg Win:         ${avg_win:.2f}")
        print(f"Avg Loss:        ${avg_loss:.2f}")
        print(f"Risk/Reward:     {abs(avg_win/avg_loss):.2f}")
        
    return trades, equity_curve

# Ejecutar
run_kaggle_backtest(DATA_PATH, final_model)
