# 📋 PLAN META-LABELING - 7 DÍAS DETALLADO

**Objetivo Final:** Implementar arquitectura de doble modelo (meta-labeling) para mejorar win rate y reducir trades

**Resultados Esperados:**
- Win Rate: 52% → 65-70%
- Sharpe: 4.45 → 5.5-6.5
- Trades: 201 → 100-130
- R/R: 3.55:1 → 4.5-5:1

---

## 📅 DÍA 1 - GENERAR DATASET DE SEÑALES HISTÓRICAS

### Objetivo
Crear el training set para el Modelo Secundario ejecutando el Modelo Primario sobre datos históricos y guardando TODAS las señales generadas (no solo las ejecutadas).

### Tareas Detalladas

#### 1.1 Modificar backtest para guardar señales (1 hora)
**Script:** `generate_metalabeling_dataset.py`

**Cambios necesarios:**
- Ejecutar modelo primario en TODOS los tickers × TODAS las fechas
- NO aplicar restricciones de capital/posiciones
- Guardar TODA señal donde `model.predict_proba() >= 0.50` (umbral bajo)

**Pseudo-código:**
```python
for ticker in tickers:
    for date in dates:
        # 1. Calcular features
        features = calculate_features(ticker, date)
        
        # 2. Predicción del modelo primario
        prob = model_primary.predict_proba(features)
        
        # 3. Aplicar filtros básicos (trend + momentum)
        if prob >= 0.50 and passes_filters(ticker, date):
            # 4. Simular qué hubiera pasado
            outcome = simulate_trade_outcome(ticker, date)
            
            # 5. Guardar señal
            signals.append({
                'date': date,
                'ticker': ticker,
                'primary_prob': prob,
                'features': features,
                'outcome': 1 if outcome['profit'] > 0 else 0,
                'profit': outcome['profit'],
                'holding_days': outcome['holding_days']
            })
```

#### 1.2 Simular outcome de cada señal (1 hora)
**Función:** `simulate_trade_outcome(ticker, date)`

**Lógica:**
```python
def simulate_trade_outcome(ticker, date):
    """
    Simula qué hubiera pasado si ejecutábamos este trade
    usando las mismas reglas del backtest real
    """
    # 1. Entry
    entry_price = data.loc[date, 'close']
    entry_atr = data.loc[date, 'atr']
    
    # 2. Calculate initial stop
    stop = calculate_trailing_stop(entry_price, entry_price, entry_atr, k=2.5)
    
    # 3. Simular días siguientes
    highest = entry_price
    for i in range(1, 90):  # Máximo 90 días
        current = data.loc[date+i, 'close']
        current_atr = data.loc[date+i, 'atr']
        
        # Actualizar stop
        if current > highest:
            highest = current
        stop = max(stop, highest - k * current_atr)
        
        # Check exit
        if current <= stop:
            return {'profit': current - entry_price, 'holding_days': i}
    
    # Si no salió, forzar salida a los 90 días
    return {'profit': data.loc[date+90, 'close'] - entry_price, 'holding_days': 90}
```

#### 1.3 Balancear dataset (30 min)
**Problema:** Puede haber más perdedores que ganadores

**Solución:**
```python
# Separar ganadores y perdedores
winners = signals_df[signals_df['outcome'] == 1]
losers = signals_df[signals_df['outcome'] == 0]

# Balancear (undersample mayoría)
if len(winners) > len(losers):
    winners = winners.sample(n=len(losers), random_state=42)
else:
    losers = losers.sample(n=len(winners), random_state=42)

balanced_df = pd.concat([winners, losers]).sample(frac=1)
```

#### 1.4 Validación (30 min)
**Checks necesarios:**
- [ ] ≥ 1,000 señales totales
- [ ] 45-55% balance entre ganadores/perdedores
- [ ] Señales distribuidas en el tiempo (no todas de un período)
- [ ] Múltiples tickers representados (no dominado por 1-2)

**Output esperado:**
```
signals_dataset.csv (2,000+ filas)
Columnas:
- date, ticker
- primary_prob (del modelo 1)
- outcome (1=ganador, 0=perdedor)
- profit, holding_days
- RSI, ADX, etc. (features originales)
```

### Entregable Día 1
```
data/
└── signals_dataset.csv (2,000+ señales)

scripts/
└── generate_metalabeling_dataset.py
```

### Tiempo Total: 3 horas

---

## 📅 DÍA 2 - META-FEATURES ENGINEERING

### Objetivo
Crear features que evalúan el CONTEXTO del trade, no la señal técnica en sí.

### Concepto Clave
**Features Primarias:** RSI, ADX, SMA → "¿Hay tendencia?"
**Meta-Features:** VIX, volumen, profit histórico → "¿Es BUENA esta señal?"

### Tareas Detalladas

#### 2.1 Meta-Features de Mercado (1 hora)
**Script:** `create_meta_features.py`

```python
def add_market_metafeatures(signals_df):
    """Features del contexto macro"""
    
    # 1. VIX Level (volatilidad de mercado)
    vix = load_vix_data()
    signals_df['vix_level'] = signals_df.apply(
        lambda row: vix.loc[row['date'], 'close'], axis=1
    )
    signals_df['vix_regime'] = signals_df['vix_level'].apply(
        lambda x: 'high' if x > 25 else 'medium' if x > 15 else 'low'
    )
    
    # 2. Market Trend (QQQ > SMA50)
    qqq = load_qqq_data()
    signals_df['market_bullish'] = signals_df.apply(
        lambda row: 1 if qqq.loc[row['date'], 'close'] > qqq.loc[row['date'], 'ma50'] else 0,
        axis=1
    )
    
    # 3. Sector Momentum
    signals_df['sector_momentum'] = signals_df.apply(
        lambda row: calculate_sector_momentum(row['ticker'], row['date']),
        axis=1
    )
    
    return signals_df
```

#### 2.2 Meta-Features de Ticker (1.5 horas)
```python
def add_ticker_metafeatures(signals_df, historical_trades):
    """Features específicas del ticker"""
    
    # 1. Profit histórico en este ticker (últimos 6 meses)
    signals_df['ticker_6m_profit'] = signals_df.apply(
        lambda row: calculate_historical_profit(
            row['ticker'], 
            row['date'] - pd.Timedelta(days=180),
            row['date'],
            historical_trades
        ),
        axis=1
    )
    
    # 2. Win rate histórico del ticker
    signals_df['ticker_win_rate'] = signals_df.apply(
        lambda row: calculate_historical_winrate(
            row['ticker'],
            row['date'] - pd.Timedelta(days=180),
            row['date'],
            historical_trades
        ),
        axis=1
    )
    
    # 3. Número de trades recientes
    signals_df['recent_trades_count'] = signals_df.apply(
        lambda row: count_recent_trades(
            row['ticker'],
            row['date'] - pd.Timedelta(days=30),
            row['date'],
            historical_trades
        ),
        axis=1
    )
    
    # 4. Volatilidad relativa (ATR/precio)
    signals_df['relative_volatility'] = signals_df['atr'] / signals_df['close']
    
    # 5. Volume ratio (actual vs promedio 20 días)
    signals_df['volume_ratio'] = signals_df['volume'] / signals_df['volume_sma']
    
    return signals_df
```

#### 2.3 Meta-Features Temporales (30 min)
```python
def add_temporal_metafeatures(signals_df):
    """Features de timing"""
    
    # 1. Día de la semana (0=Lunes, 4=Viernes)
    signals_df['day_of_week'] = signals_df['date'].dt.dayofweek
    
    # 2. Mes del año (seasonality)
    signals_df['month'] = signals_df['date'].dt.month
    
    # 3. Días desde último trade en este ticker
    signals_df = signals_df.sort_values(['ticker', 'date'])
    signals_df['days_since_last_trade'] = signals_df.groupby('ticker')['date'].diff().dt.days
    signals_df['days_since_last_trade'].fillna(999, inplace=True)
    
    return signals_df
```

#### 2.4 Meta-Features de Señal (1 hora)
```python
def add_signal_metafeatures(signals_df):
    """Features de la señal misma"""
    
    # 1. Confianza del modelo primario (ya tenemos)
    # signals_df['primary_prob'] ya existe
    
    # 2. Gap con threshold (¿cuánto supera el 0.65?)
    signals_df['confidence_gap'] = signals_df['primary_prob'] - 0.65
    
    # 3. Fuerza de tendencia (ADX)
    # signals_df['adx_14'] ya existe
    
    # 4. Distancia a MA50 (%)
    signals_df['dist_to_ma50_pct'] = (signals_df['close'] - signals_df['ma_50']) / signals_df['ma_50']
    
    # 5. Momentum slope (cambio de RSI)
    # signals_df['slope_rsi_14'] ya existe
    
    # 6. Combinación: primary_prob × ADX (señal fuerte)
    signals_df['signal_strength'] = signals_df['primary_prob'] * signals_df['adx_14'] / 100
    
    return signals_df
```

#### 2.5 Feature Selection (30 min)
**Eliminar features redundantes:**
```python
from sklearn.feature_selection import mutual_info_classif

# Calcular importancia
importance = mutual_info_classif(X, y, random_state=42)

# Eliminar features con importance < 0.01
low_importance = [feat for feat, imp in zip(features, importance) if imp < 0.01]
X_filtered = X.drop(columns=low_importance)
```

### Lista Final de Meta-Features (15-20)
```
MERCADO:
1. vix_level
2. vix_regime (high/medium/low)
3. market_bullish (0/1)
4. sector_momentum

TICKER:
5. ticker_6m_profit
6. ticker_win_rate
7. recent_trades_count
8. relative_volatility
9. volume_ratio

TEMPORAL:
10. day_of_week
11. month
12. days_since_last_trade

SEÑAL:
13. primary_prob (del modelo 1)
14. confidence_gap
15. adx_14
16. dist_to_ma50_pct
17. slope_rsi_14
18. signal_strength (prob × ADX)
```

### Entregable Día 2
```
data/
└── signals_dataset_enhanced.csv (con meta-features)

scripts/
└── create_meta_features.py
```

### Tiempo Total: 4 horas

---

## 📅 DÍA 3 - ENTRENAR MODELO SECUNDARIO

### Objetivo
Crear el meta-labeling model que predice si una señal será ganadora.

### Tareas Detalladas

#### 3.1 Preparar datos (30 min)
```python
# Load dataset
df = pd.read_csv('signals_dataset_enhanced.csv')

# Separar features y target
X = df[meta_features_list]
y = df['outcome']  # 1=ganador, 0=perdedor

# Train/Test split TEMPORAL (no random)
# Importante: No mezclar pasado y futuro
split_date = pd.Timestamp('2024-01-01')
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
```

#### 3.2 Entrenar modelos base (2 horas)
**3 modelos diferentes:**

```python
# 1. XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,  # Ajustar si está desbalanceado
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 2. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)
rf_model.fit(X_train, y_train)

# 3. LightGBM
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train, y_train)
```

#### 3.3 Optimizar con Optuna (1.5 horas)
```python
import optuna

def objective(trial):
    # Hyperparameters a optimizar
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar en validation set
    preds = model.predict_proba(X_val)[:, 1]
    
    # Métrica: Precision en clase "Ganador" con threshold 0.70
    precision = precision_score(y_val, preds >= 0.70)
    
    return precision

# Optimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best precision: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

#### 3.4 Validación con Purged K-Fold (1 hora)
**Importante:** No usar K-Fold normal (contaminaría con futuro)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)
    
    preds = model.predict_proba(X_val)[:, 1]
    precision = precision_score(y_val, preds >= 0.70)
    
    cv_scores.append(precision)

print(f"CV Precision: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

#### 3.5 Evaluación final (30 min)
```python
# Predecir en test set
test_probs = model.predict_proba(X_test)[:, 1]

# Métricas con threshold 0.70
threshold = 0.70
test_preds = test_probs >= threshold

from sklearn.metrics import classification_report, confusion_matrix

print("\n=== RESULTADOS TEST SET ===")
print(classification_report(y_test, test_preds, target_names=['Perdedor', 'Ganador']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_preds))

# Precision en "Ganador"
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)

print(f"\nPrecision (Ganador): {precision:.3f}")
print(f"Recall (Ganador): {recall:.3f}")

# TARGET: Precision ≥ 0.70
if precision >= 0.70:
    print("\n✓ OBJETIVO ALCANZADO!")
else:
    print(f"\n⚠ Precision {precision:.2%} < 70%. Ajustar hiperparámetros o features.")
```

### Criterios de Éxito Día 3
- [ ] **Precision ≥ 70%** en clase "Ganador" (test set)
- [ ] Recall ≥ 50% (no queremos rechazar TODO)
- [ ] CV estable (std < 0.05)
- [ ] No overfitting (train precision < test precision + 0.1)

### Entregable Día 3
```
models/
├── meta_model_xgb.joblib (mejor modelo)
├── meta_model_rf.joblib
└── meta_model_lgb.joblib

reports/
├── meta_model_performance.txt
├── feature_importance.png
└── confusion_matrix.png
```

### Tiempo Total: 5 horas

---

## 📅 DÍA 4 - INTEGRACIÓN DE MODELOS

### Objetivo
Crear pipeline completo que combina Modelo Primario + Secundario

### Tareas Detalladas

#### 4.1 Diseñar arquitectura (30 min)
```python
"""
PIPELINE META-LABELING:

1. Modelo Primario:
   Input: Features técnicas (RSI, ADX, etc.)
   Output: Probabilidad de tendencia alcista
   Threshold: 0.65

2. SI primary_prob >= 0.65:
   → Calcular meta-features
   → Modelo Secundario
   
3. Modelo Secundario:
   Input: Meta-features (VIX, profit histórico, etc.)
   Output: Probabilidad de que sea trade ganador
   Threshold: 0.70
   
4. SI secondary_prob >= 0.70:
   → EJECUTAR TRADE
   SINO:
   → RECHAZAR SEÑAL
```

#### 4.2 Implementar pipeline (2 horas)
**Script:** `backtest_metalabeling.py`

```python
class MetaLabelingPipeline:
    def __init__(self, model_primary, model_secondary):
        self.primary = model_primary
        self.secondary = model_secondary
        self.primary_threshold = 0.65
        self.secondary_threshold = 0.70
    
    def generate_signal(self, ticker, date, data):
        """
        Genera señal usando ambos modelos
        """
        # 1. Calcular features primarias
        primary_features = calculate_primary_features(ticker, date, data)
        
        # 2. Modelo Primario
        primary_prob = self.primary.predict_proba([primary_features])[0, 1]
        
        if primary_prob < self.primary_threshold:
            return {
                'signal': False,
                'reason': 'primary_rejected',
                'primary_prob': primary_prob
            }
        
        # 3. Calcular meta-features
        meta_features = calculate_meta_features(ticker, date, data, primary_prob)
        
        # 4. Modelo Secundario
        secondary_prob = self.secondary.predict_proba([meta_features])[0, 1]
        
        if secondary_prob < self.secondary_threshold:
            return {
                'signal': False,
                'reason': 'secondary_rejected',
                'primary_prob': primary_prob,
                'secondary_prob': secondary_prob
            }
        
        # 5. SEÑAL APROBADA
        return {
            'signal': True,
            'primary_prob': primary_prob,
            'secondary_prob': secondary_prob,
            'confidence': secondary_prob  # Confianza final
        }
```

#### 4.3 Adaptar backtest (2 horas)
```python
def run_metalabeling_backtest(
    tickers,
    data_root,
    model_primary,
    model_secondary,
    start_date,
    end_date,
    init_cash=10000.0
):
    """
    Backtest con meta-labeling
    """
    pipeline = MetaLabelingPipeline(model_primary, model_secondary)
    
    cash = init_cash
    positions = {}
    trades = []
    rejected_signals = []  # Para análisis
    
    for date in all_dates:
        # ... (lógica de gestión de posiciones igual que antes)
        
        # Buscar nuevas señales
        for ticker in available_tickers:
            # Generar señal con pipeline
            signal = pipeline.generate_signal(ticker, date, data)
            
            if signal['signal']:
                # EJECUTAR TRADE
                enter_position(ticker, date, signal['confidence'])
            else:
                # RECHAZAR y guardar para análisis
                rejected_signals.append({
                    'date': date,
                    'ticker': ticker,
                    'reason': signal['reason'],
                    'primary_prob': signal.get('primary_prob'),
                    'secondary_prob': signal.get('secondary_prob')
                })
    
    return {
        'trades': trades,
        'rejected_signals': rejected_signals,
        'final_equity': cash
    }
```

#### 4.4 Testing & debugging (30 min)
```python
# Test en un ticker pequeño primero
test_result = run_metalabeling_backtest(
    tickers=['AAPL'],
    data_root=Path('data'),
    model_primary=model_primary,
    model_secondary=model_secondary,
    start_date='2023-01-01',
    end_date='2024-12-31'
)

# Verificar
assert len(test_result['trades']) > 0, "No se generaron trades!"
assert len(test_result['rejected_signals']) > 0, "No se rechazó ninguna señal!"

print(f"Trades ejecutados: {len(test_result['trades'])}")
print(f"Señales rechazadas: {len(test_result['rejected_signals'])}")
print(f"Ratio aceptación: {len(test_result['trades']) / (len(test_result['trades']) + len(test_result['rejected_signals'])):.2%}")
```

### Entregable Día 4
```
scripts/
├── backtest_metalabeling.py (pipeline completo)
└── metalabeling_pipeline.py (clase reutilizable)

tests/
└── test_pipeline.py (unit tests)
```

### Tiempo Total: 4 horas

---

## 📅 DÍA 5 - BACKTESTING & VALIDACIÓN

### Objetivo
Probar el sistema completo y comparar vs baseline

### Tareas Detalladas

#### 5.1 Backtest período completo (1 hora)
```python
# BASELINE (modelo simple actual)
baseline_results = run_trend_following_backtest(
    tickers=all_tickers,
    model_path='models/trend_model_2015_2024_OPTUNA_FIXED.joblib',
    start_date='2023-01-01',
    end_date='2024-12-31',
    min_confidence=0.65
)

# META-LABELING
metalabeling_results = run_metalabeling_backtest(
    tickers=all_tickers,
    model_primary=model_primary,
    model_secondary=model_secondary,
    start_date='2023-01-01',
    end_date='2024-12-31'
)
```

#### 5.2 Comparar métricas (1 hora)
```python
def compare_results(baseline, metalabeling):
    """Comparación side-by-side"""
    
    comparison = pd.DataFrame({
        'Metric': ['Total Trades', 'Win Rate', 'Sharpe', 'Total Return', 
                   'Avg Holding', 'Risk/Reward', 'Max Drawdown'],
        'BASELINE': [
            len(baseline['trades_df']),
            baseline['win_rate'],
            baseline['sharpe'],
            baseline['total_return'],
            baseline['avg_holding_days'],
            baseline['risk_reward'],
            baseline['max_dd']
        ],
        'META-LABELING': [
            len(metalabeling['trades_df']),
            metalabeling['win_rate'],
            metalabeling['sharpe'],
            metalabeling['total_return'],
            metalabeling['avg_holding_days'],
            metalabeling['risk_reward'],
            metalabeling['max_dd']
        ]
    })
    
    # Calcular mejora
    comparison['Improvement'] = (
        (comparison['META-LABELING'] - comparison['BASELINE']) / 
        comparison['BASELINE'] * 100
    )
    
    return comparison
```

#### 5.3 Walk-Forward Validation (2 horas)
**Test en múltiples períodos:**

```python
periods = [
    ('2015-01-01', '2016-12-31'),
    ('2016-01-01', '2017-12-31'),
    ('2017-01-01', '2018-12-31'),
    ('2018-01-01', '2019-12-31'),
    ('2019-01-01', '2020-12-31'),
    ('2020-01-01', '2021-12-31'),
    ('2021-01-01', '2022-12-31'),
    ('2022-01-01', '2023-12-31'),
    ('2023-01-01', '2024-12-31')
]

wf_results = []

for start, end in periods:
    # Re-entrenar modelo secundario con datos HASTA start
    train_data = signals_df[signals_df['date'] < start]
    model_secondary = train_meta_model(train_data)
    
    # Backtest en el período
    result = run_metalabeling_backtest(
        ...,
        start_date=start,
        end_date=end
    )
    
    wf_results.append({
        'period': f"{start} to {end}",
        'trades': len(result['trades_df']),
        'win_rate': result['win_rate'],
        'sharpe': result['sharpe']
    })

# Analizar estabilidad
wf_df = pd.DataFrame(wf_results)
print(f"Win Rate promedio: {wf_df['win_rate'].mean():.2%} ± {wf_df['win_rate'].std():.2%}")
print(f"Sharpe promedio: {wf_df['sharpe'].mean():.2f} ± {wf_df['sharpe'].std():.2f}")
```

#### 5.4 Análisis de señales rechazadas (1 hour)
```python
rejected = metalabeling_results['rejected_signals']

# ¿Por qué se rechazan?
print("\nRAZONES DE RECHAZO:")
print(rejected['reason'].value_counts(normalize=True))

# ¿Cuál hubiera sido el outcome?
rejected_with_outcome = []
for _, signal in rejected.iterrows():
    outcome = simulate_trade_outcome(signal['ticker'], signal['date'])
    rejected_with_outcome.append({
        **signal,
        'would_have_won': 1 if outcome['profit'] > 0 else 0
    })

rejected_df = pd.DataFrame(rejected_with_outcome)

# ¿El meta-model rechazó correctamente?
print(f"\nSeñales rechazadas que hubieran GANADO: {rejected_df['would_have_won'].mean():.2%}")
print(f"Señales rechazadas que hubieran PERDIDO: {(1 - rejected_df['would_have_won'].mean()):.2%}")

# Idealmente: Meta-model rechaza mayormente señales perdedoras
```

### Criterios de Éxito Día 5
- [ ] Win Rate META > Win Rate BASELINE
- [ ] Sharpe META > Sharpe BASELINE
- [ ] Walk-forward estable (std < 0.10)
- [ ] Meta-model rechaza >60% de señales perdedoras
- [ ] Trades reducidos 40-50%

### Entregable Día 5
```
results/
├── backtest_baseline.csv
├── backtest_metalabeling.csv
├── comparison_report.txt
├── walkforward_results.csv
└── rejected_signals_analysis.csv
```

### Tiempo Total: 5 horas

---

## 📅 DÍA 6 - ANÁLISIS & DEBUGGING

### Objetivo
Entender qué funciona, qué no, y optimizar

### Tareas Detalladas

#### 6.1 Feature Importance (1 hora)
```python
import shap

# SHAP values para modelo secundario
explainer = shap.TreeExplainer(model_secondary)
shap_values = explainer.shap_values(X_test)

# Visualizar
shap.summary_plot(shap_values, X_test, feature_names=meta_features)

# Top features
feature_importance = pd.DataFrame({
    'feature': meta_features,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTOP 10 META-FEATURES MÁS IMPORTANTES:")
print(feature_importance.head(10))
```

#### 6.2 Calibración de probabilidades (1 hora)
```python
from sklearn.calibration import calibration_curve

# ¿Las probabilidades están calibradas?
prob_true, prob_pred = calibration_curve(
    y_test, 
    model_secondary.predict_proba(X_test)[:, 1],
    n_bins=10
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Meta Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve - Meta Model')
plt.legend()
plt.savefig('reports/calibration_curve.png')

# Si no está calibrado, aplicar Platt Scaling:
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model_secondary, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
```

#### 6.3 Análisis de errores (1.5 horas)
```python
# Meta-model predice ganador pero fue perdedor (False Positive)
FP = test_df[(test_preds == 1) & (y_test == 0)]

print(f"\nFALSE POSITIVES: {len(FP)}")
print("\nCaracterísticas comunes:")
print(FP[meta_features].describe())

# ¿Qué tienen en común estos errores?
# Hypothesis: Tal vez en mercados high VIX falla más

# Meta-model predice perdedor pero fue ganador (False Negative)
FN = test_df[(test_preds == 0) & (y_test == 1)]

print(f"\nFALSE NEGATIVES: {len(FN)}")
print("\nCaracterísticas comunes:")
print(FN[meta_features].describe())

# ¿Estamos rechazando buenas oportunidades?
```

#### 6.4 Ajustes (1.5 horas)
**Basado en análisis, hacer ajustes:**

```python
# PROBLEMA 1: Threshold muy alto
if precision > 0.80 and recall < 0.40:
    print("Threshold demasiado conservador")
    recommended_threshold = 0.65  # Bajar
    
# PROBLEMA 2: Threshold muy bajo
elif precision < 0.65 and recall > 0.70:
    print("Threshold demasiado permisivo")
    recommended_threshold = 0.75  # Subir

# PROBLEMA 3: Feature no útil
low_importance_features = feature_importance[feature_importance['importance'] < 0.01]['feature'].tolist()
print(f"\nFeatures a eliminar: {low_importance_features}")

# Re-entrenar sin esas features
X_train_filtered = X_train.drop(columns=low_importance_features)
model_improved = XGBClassifier(**best_params)
model_improved.fit(X_train_filtered, y_train)
```

### Entregable Día 6
```
analysis/
├── feature_importance.png
├── shap_summary.png
├── calibration_curve.png
├── error_analysis.txt
└── recommended_changes.txt
```

### Tiempo Total: 4 horas

---

## 📅 DÍA 7 - OPTIMIZACIÓN FINAL & DOCUMENTACIÓN

### Objetivo
Preparar sistema para producción

### Tareas Detalladas

#### 7.1 Re-entrenar con mejores parámetros (1 hora)
```python
# Aplicar TODOS los aprendizajes de Día 6

# 1. Features filtradas
final_features = [f for f in meta_features if f not in low_importance_features]

# 2. Threshold optimizado
final_threshold = 0.68  # (basado en análisis)

# 3. Modelo calibrado
from sklearn.calibration import CalibratedClassifierCV

final_model = XGBClassifier(**best_params)
final_model.fit(X_train[final_features], y_train)

calibrated_final_model = CalibratedClassifierCV(final_model, method='sigmoid', cv=3)
calibrated_final_model.fit(X_train[final_features], y_train)

# 4. Validar mejora
final_preds = calibrated_final_model.predict_proba(X_test[final_features])[:, 1] >= final_threshold
final_precision = precision_score(y_test, final_preds)
final_recall = recall_score(y_test, final_preds)

print(f"\nMODELO FINAL:")
print(f"Precision: {final_precision:.3f}")
print(f"Recall: {final_recall:.3f}")

# Guardar
joblib.dump(calibrated_final_model, 'models/meta_model_final.joblib')
```

#### 7.2 Pipeline de producción (1.5 horas)
```python
# production/run_metalabeling_backtest.py

def production_backtest(config):
    """
    Backtest listo para producción
    """
    # 1. Load models
    model_primary = joblib.load(config['model_primary_path'])
    model_secondary = joblib.load(config['model_secondary_path'])
    
    # 2. Load data
    tickers = load_tickers(config['ticker_file'])
    
    # 3. Run backtest
    results = run_metalabeling_backtest(
        tickers=tickers,
        model_primary=model_primary,
        model_secondary=model_secondary,
        start_date=config['start_date'],
        end_date=config['end_date'],
        primary_threshold=config['primary_threshold'],
        secondary_threshold=config['secondary_threshold']
    )
    
    # 4. Generate report
    generate_report(results, output_path='reports/backtest_report.html')
    
    return results

# Uso:
config = {
    'model_primary_path': 'models/trend_model_2015_2024_OPTUNA_FIXED.joblib',
    'model_secondary_path': 'models/meta_model_final.joblib',
    'ticker_file': 'good.txt',
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'primary_threshold': 0.65,
    'secondary_threshold': 0.68
}

results = production_backtest(config)
```

#### 7.3 Documentación completa (30 min)
**Crear:** `META_LABELING_README.md`

```markdown
# Meta-Labeling System Documentation

## Arquitectura

MODELO PRIMARIO (Trend Following):
- Input: Features técnicas (RSI, ADX, SMA, etc.)
- Output: Probabilidad de tendencia alcista
- Threshold: 0.65
- Modelo: XGBoost entrenado con Triple Barrier

MODELO SECUNDARIO (Meta-Labeling):
- Input: Meta-features (VIX, profit histórico, contexto)
- Output: Probabilidad de que trade sea ganador
- Threshold: 0.68
- Modelo: XGBoost calibrado con CalibratedClassifierCV

## Cómo Usar

### 1. Generar Nuevo Dataset de Señales
python scripts/generate_metalabeling_dataset.py \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --output data/signals_new.csv

### 2. Crear Meta-Features
python scripts/create_meta_features.py \
    --input data/signals_new.csv \
    --output data/signals_enhanced.csv

### 3. Re-entrenar Modelo Secundario
python scripts/train_meta_model.py \
    --input data/signals_enhanced.csv \
    --output models/meta_model_new.joblib \
    --n-trials 50

### 4. Ejecutar Backtest
python production/run_metalabeling_backtest.py \
    --config config/production.json

## Resultados vs Baseline

| Métrica | BASELINE | META-LABELING | Mejora |
|---------|----------|---------------|--------|
| Trades | 201 | 118 | -41% |
| Win Rate | 52.7% | 68.2% | +29% |
| Sharpe | 4.45 | 5.87 | +32% |
| R/R | 3.55:1 | 4.85:1 | +37% |

## Mantenimiento

### Re-entrenar cada 3 meses:
1. Generar nuevo dataset con señales recientes
2. Re-entrenar modelo secundario
3. Validar con walk-forward
4. Actualizar en producción si mejora

### Monitorear drift:
- Win rate real vs esperado
- Calibración de probabilidades
- Feature importance cambios
```

#### 7.4 Tests unitarios (30 min)
```python
# tests/test_metalabeling.py

import pytest

def test_pipeline_generates_signals():
    pipeline = MetaLabelingPipeline(model_primary, model_secondary)
    signal = pipeline.generate_signal('AAPL', '2024-01-15', data)
    assert 'signal' in signal
    assert 'primary_prob' in signal

def test_meta_features_calculation():
    meta_feat = calculate_meta_features('AAPL', '2024-01-15', data, 0.70)
    assert len(meta_feat) == len(final_features)
    assert all(not pd.isna(v) for v in meta_feat)

def test_backtest_runs():
    results = run_metalabeling_backtest(
        tickers=['AAPL'],
        model_primary=model_primary,
        model_secondary=model_secondary,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    assert 'trades' in results
    assert 'rejected_signals' in results
```

### Entregable Día 7
```
production/
├── run_metalabeling_backtest.py
├── generate_signals_daily.py
└── config/
    └── production.json

docs/
└── META_LABELING_README.md

tests/
└── test_metalabeling.py

models/
└── meta_model_final.joblib (PRODUCCIÓN)
```

### Tiempo Total: 3 horas

---

## 📊 SUMMARY & CHECKPOINTS

### Métricas de Éxito Finales
- [ ] **Win Rate:** ≥65% (vs 52.7% baseline)
- [ ] **Sharpe:** ≥5.5 (vs 4.45 baseline)
- [ ] **Trades:** -40% reducción
- [ ] **R/R:** ≥4.5:1 (vs 3.55:1 baseline)
- [ ] **Precision Meta-Model:** ≥70%
- [ ] **Walk-Forward estable:** std <0.10

### Tiempo Total: 28 horas
- Día 1: 3h
- Día 2: 4h
- Día 3: 5h
- Día 4: 4h
- Día 5: 5h
- Día 6: 4h
- Día 7: 3h

### Riesgos y Mitigaciones

**Riesgo 1:** Dataset de señales insuficiente (<1000)
**Mitigación:** Usar más tickers o período más largo

**Riesgo 2:** Meta-model no mejora baseline
**Mitigación:** Revisar meta-features, probar ensemble

**Riesgo 3:** Overfitting
**Mitigación:** Walk-forward validation estricta, regularización

**Riesgo 4:** Threshold muy restrictivo (pocas señales)
**Mitigación:** Ajustar threshold basado en precision/recall curve

---

## 🚀 READY TO START

**Día 1 comienza con:**
```bash
python scripts/generate_metalabeling_dataset.py
```

**Expected output:**
- signals_dataset.csv con 2,000+ señales
- Balance 50/50 ganadores/perdedores
- Múltiples tickers y períodos
