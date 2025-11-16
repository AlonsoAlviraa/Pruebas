# =============================================================================
# SCRIPT CONSOLIDADO: ESTRATEGIA DE TRADING BASADA EN EVENTOS
# =============================================================================

# --- [Sección de Importaciones] ---
# Se agrupan todas las importaciones requeridas por el notebook
from newsapi import NewsApiClient
from textblob import TextBlob
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- [CELDA 1: Comentarios Introductorios] ---
# Why would we want to analyze event-driven trading strategies?
# Capitalize on Market Inefficiencies
# Generate Alpha
# Diversify Return Sources
# Quantify News and Sentiment Impact
# Backtest and Validate Trading Hypotheses
# Risk Management
# Automate Decision-Making
# Build Event-Driven Products

# Component	                  Description
# Data Source	              Yahoo Finance News, NewsAPI, Econ releases (FRED, Bloomberg headlines, etc)
# NLP Processor	              Sentiment and event trigger extractor (e.g. TextBlob, spaCy, transformers)
# Event Detector	          Checks for specific keywords like “beat earnings”, “Fed hike”, “job growth”
# Trade Logic Engine	      If X is true, allocate Y; based on strategy rules
# Execution Engine	          Paper trading via Alpaca or live broker (e.g. Interactive Brokers API)


# --- [CELDA 2: News + Economic Event Feeds] ---
print("\n--- [CELDA 2: Obteniendo Artículos de Noticias] ---")

# --- ¡IMPORTANTE! ---
# Reemplace 'TU_API_KEY_AQUI' con su clave de API real de NewsApiClient
API_KEY = '81565b75c4d74b21aa9092bd445e732f' 
# -------------------

if API_KEY == 'TU_API_KEY_AQUI':
    print("ADVERTENCIA: Por favor, reemplace 'TU_API_KEY_AQUI' con su clave de NewsAPI en el script.")
    # Se crea una lista de artículos ficticia para permitir que el script continúe
    # En un escenario real, esto debería detenerse o lanzar un error.
    articles = [
        {'title': 'Apple earnings beat expectations', 'description': 'Strong iPhone sales boost AAPL.', 'url': 'http://example.com/aapl', 'publishedAt': '2025-07-30T12:00:00Z'},
        {'title': 'Fed signals rate hike', 'description': 'Inflation concerns grow, market reacts negatively.', 'url': 'http://example.com/fed', 'publishedAt': '2025-07-30T13:00:00Z'},
        {'title': 'Microsoft misses earnings', 'description': 'MSFT stock drops on weak guidance.', 'url': 'http://example.com/msft', 'publishedAt': '2025-07-30T14:00:00Z'},
        {'title': 'Amazon new jobs report', 'description': 'AMZN to hire 50,000 new workers.', 'url': 'http://example.com/amzn', 'publishedAt': '2025-07-31T10:00:00Z'},
        {'title': 'Google CPI analysis', 'description': 'GOOGL data shows inflation stabilizing.', 'url': 'http://example.com/googl', 'publishedAt': '2025-07-31T11:00:00Z'}
    ]
else:
    newsapi = NewsApiClient(api_key=API_KEY)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    articles = []

    for ticker in tickers:
        try:
            res = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=100)
            articles.extend(res.get('articles', []))
            print(f"Artículos obtenidos para {ticker}: {len(res.get('articles', []))}")
        except Exception as e:
            print(f"Error al obtener noticias para {ticker}: {e}")
            # Esto puede ocurrir si la API key es inválida o se excede el límite
            if 'apiKeyInvalid' in str(e):
                print("Error fatal: La clave API es inválida.")
                exit() # Detener el script si la clave es mala

    print(f"Total de artículos obtenidos: {len(articles)}")


# --- [CELDA 3: NLP-Based Event Detection] ---
print("\n--- [CELDA 3: Procesamiento NLP y Detección de Eventos] ---")

# Define your tickers and related company keywords
ticker_keywords = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "GOOGL": ["google", "alphabet"],
    "AMZN": ["amazon"],
    "META": ["meta", "facebook"],
    "TSLA": ["tesla", "elon"]
}

# Keywords that trigger event-driven logic
event_keywords = ["earnings", "fed", "inflation", "cpi", "jobs", "fomc", "rate hike", "interest rate", "guidance", "announcement"]

def extract_sentiment_and_event(article):
    title = article.get('title', '') or '' # Asegura que no sea None
    content = article.get('description', '') or '' # Asegura que no sea None
    text = (title + " " + content).lower()

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    event_detected = any(keyword in text for keyword in event_keywords)

    matched_tickers = [
        ticker for ticker, aliases in ticker_keywords.items()
        if any(alias in text for alias in aliases)
    ]

    # Try to get the article's publish timestamp
    published_at = article.get('publishedAt') or article.get('published_at') or article.get('pubDate') or article.get('date') or ''

    return {
        "tickers": matched_tickers,
        "sentiment": sentiment,
        "event_trigger": event_detected,
        "title": title,
        "url": article.get('url', ''),
        "timestamp": published_at  
    }

# Apply to all articles
events = [extract_sentiment_and_event(a) for a in articles]

# Explode multi-ticker rows into individual rows
df_events = pd.DataFrame(events)
df_events = df_events.explode("tickers").dropna(subset=["tickers"])
df_events = df_events[df_events["event_trigger"] == True]

# Ensure timestamp column is datetime
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], errors='coerce', utc=True)
df_events = df_events.dropna(subset=['timestamp'])

if not df_events.empty:
    df_events = df_events.set_index('timestamp').sort_index()

    # Only use events that match dates for which we have price data
    event_start = df_events.index.min().date() - pd.Timedelta(days=10)
    event_end = df_events.index.max().date() + pd.Timedelta(days=1)
    dates = pd.date_range(event_start, event_end, tz="UTC")

    print(f"Inicio del rango de eventos: {event_start}")
    print(f"Fin del rango de eventos: {event_end}")

    # Format
    df_events = df_events[["tickers", "sentiment", "title", "url"]].rename(columns={"tickers": "ticker"})

    print(f"Forma del DataFrame de eventos: {df_events.shape}")
    print("\nHead del DataFrame de Eventos:")
    print(df_events.head(10))
    print("\nTail del DataFrame de Eventos:")
    print(df_events.tail(10))
else:
    print("No se encontraron eventos relevantes con la API key de prueba o los datos obtenidos.")
    # Crear un DataFrame vacío con las columnas esperadas para evitar errores posteriores
    df_events = pd.DataFrame(columns=["ticker", "sentiment", "title", "url", "signal"])


# --- [CELDA 4: Trade Decision Logic] ---
print("\n--- [CELDA 4: Asignando Lógica de Decisión] ---")

def trade_signal(row):
    if row['sentiment'] > 0.20:
        return "BUY"
    elif row['sentiment'] < -0.20:
        return "SELL"
    else:
        return "HOLD"

if not df_events.empty:
    df_events['signal'] = df_events.apply(trade_signal, axis=1)
    print("\nHead del DataFrame de Eventos con Señales:")
    print(df_events.head(10))
else:
    print("DataFrame de eventos vacío, saltando la asignación de señales.")


# --- [CELDA 5: Resumen de Tickers] ---
print("\n--- [CELDA 5: Resumen de Sentimiento por Ticker] ---")

if not df_events.empty:
    summary = df_events.groupby("ticker")["sentiment"].agg(["count", "mean"]).reset_index()
    summary.columns = ["Ticker", "EventCount", "AvgSentiment"]
    summary = summary.sort_values(by=['AvgSentiment'], ascending=False)
    summary = summary.sort_values(by=['EventCount'], ascending=False)
    
    print(summary)
else:
    print("DataFrame de eventos vacío, no se puede generar resumen.")


# --- [CELDA 6: Simulación de Cartera (Conceptual)] ---
print("\n--- [CELDA 6: Simulación de Cartera (Demo de Lógica)] ---")
# NOTA TÉCNICA: Esta celda usa un precio estático ('1d') para simular
# operaciones sobre eventos históricos. Es una demostración de lógica,
# no un backtest válido. El valor final será 100,000.

valid_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
initial_cash = 100_000
positions = {t: 0 for t in valid_tickers}
portfolio_value = initial_cash

# Obtener precios actuales (o del último día de mercado)
price_lookup = {}
for t in valid_tickers:
    try:
        price_lookup[t] = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
    except IndexError:
        print(f"No se pudo obtener el precio para {t}, usando 100 como placeholder.")
        price_lookup[t] = 100.0 # Placeholder si falla la API de yf

if not df_events.empty:
    for _, row in df_events.iterrows():
        ticker = row['ticker']
        signal = row['signal']
        
        # Asegurarse de que el ticker del evento esté en nuestra lista de precios
        if ticker not in price_lookup:
            continue
            
        price = price_lookup[ticker]
        
        if signal == 'BUY':
            shares = int(0.05 * portfolio_value / price)
            cost = shares * price
            if shares > 0 and portfolio_value >= cost:
                positions[ticker] += shares
                portfolio_value -= cost
                print(f"BUY {shares} of {ticker} at ${price:.2f} (Cost: ${cost:.2f})")
        elif signal == 'SELL' and positions[ticker] > 0:
            shares = positions[ticker]
            proceeds = shares * price
            portfolio_value += proceeds
            positions[ticker] = 0
            print(f"SELL {shares} of {ticker} at ${price:.2f} (Proceeds: ${proceeds:.2f})")

    final_value = portfolio_value + sum(positions[t] * price_lookup[t] for t in positions)
    print("Final Portfolio Value (simulación demo):", round(final_value, 2))
    print("Final Positions (simulación demo):", positions)
else:
    print("DataFrame de eventos vacío, saltando simulación de cartera.")


# --- [CELDA 7: Test de Umbrales de Sentimiento] ---
print("\n--- [CELDA 7: Prueba de Optimización de Umbrales] ---")

if not df_events.empty:
    thresholds = np.arange(0.05, 0.5, 0.05)
    results = []

    for t in thresholds:
        df_temp = df_events.copy()
        
        def signal_logic(row):
            if row['sentiment'] > t:
                return 'BUY'
            elif row['sentiment'] < -t:
                return 'SELL'
            else:
                return 'HOLD'
        
        df_temp['signal'] = df_temp.apply(signal_logic, axis=1)
        
        # Count trades
        buys = df_temp['signal'].value_counts().get('BUY', 0)
        sells = df_temp['signal'].value_counts().get('SELL', 0)
        
        # Score = bias toward actionable signals
        score = (buys + sells) / len(df_temp)
        results.append({'threshold': t, 'BUYs': buys, 'SELLs': sells, 'actionable_ratio': score})

    # Display results
    df_thresholds = pd.DataFrame(results)
    print(df_thresholds.sort_values('actionable_ratio', ascending=False))
else:
    print("DataFrame de eventos vacío, saltando prueba de umbrales.")


# --- [CELDA 8: Optimización de Asignación] ---
# Esta celda no estaba ejecutada en el notebook original, pero es
# necesaria para la Celda 9 (Backtesting).
print("\n--- [CELDA 8: Cálculo de Pesos de Cartera Estáticos] ---")

if 'signal' in df_events.columns:
    buy_counts = df_events[df_events['signal'] == 'BUY']['ticker'].value_counts()
    sell_counts = df_events[df_events['signal'] == 'SELL']['ticker'].value_counts()
    all_tickers = df_events['ticker'].unique()

    # Create allocation score = buys - sells
    allocation_df = pd.DataFrame(index=all_tickers)
    allocation_df['BUY'] = buy_counts
    allocation_df['SELL'] = sell_counts
    allocation_df = allocation_df.fillna(0)
    allocation_df['score'] = allocation_df['BUY'] - allocation_df['SELL']

    # Normalize scores to allocate capital
    allocation_df['score'] = allocation_df['score'].clip(lower=0)  # no negative weights
    
    if allocation_df['score'].sum() > 0:
        allocation_df['weight'] = allocation_df['score'] / allocation_df['score'].sum()
    else:
        # Asignación por igual si no hay señales de compra netas
        print("No hay puntuaciones positivas netas; se usará asignación equitativa.")
        valid_count = len(all_tickers)
        allocation_df['weight'] = 1 / valid_count if valid_count > 0 else 0

    allocation_df['allocation_$'] = (allocation_df['weight'] * 100_000).round(2)
    allocation_df = allocation_df.sort_values(by=['allocation_$'], ascending=False)
    
    print(allocation_df[['BUY', 'SELL', 'score', 'weight', 'allocation_$']])
else:
    print("Faltan señales ('signal') en df_events. Creando asignación equitativa para el backtest.")
    # Fallback para que el backtest pueda ejecutarse
    allocation_df = pd.DataFrame(index=valid_tickers)
    allocation_df['weight'] = 1.0 / len(valid_tickers) if len(valid_tickers) > 0 else 0
    allocation_df['score'] = 0
    allocation_df['BUY'] = 0
    allocation_df['SELL'] = 0
    allocation_df['allocation_$'] = (allocation_df['weight'] * 100_000).round(2)


# --- [CELDA 9: Backtest de Cartera Estática vs S&P 500] ---
print("\n--- [CELDA 9: Ejecutando Backtest (2024)] ---")

# Define parameters
initial_capital = 100_000
tickers = allocation_df.index.tolist()
weights = allocation_df["weight"].to_dict()
backtest_start = "2024-01-01"
backtest_end = "2024-12-31"

# Asegurarse de que solo se incluyan tickers válidos
tickers = [t for t in tickers if t in valid_tickers]
# Re-balancear pesos si algunos tickers fueron filtrados
total_weight = sum(weights.get(t, 0) for t in tickers)
if total_weight == 0 and len(tickers) > 0: # Fallback si los pesos son 0
    weights = {t: 1.0/len(tickers) for t in tickers}
    total_weight = 1.0
elif total_weight < 1.0 and total_weight > 0:
    weights = {t: weights.get(t, 0) / total_weight for t in tickers}

print(f"Tickers para Backtest: {tickers}")
print(f"Pesos para Backtest: {weights}")

if tickers:
    # Step 1: Download historical prices
    price_data = yf.download(tickers + ["^GSPC"], start=backtest_start, end=backtest_end)["Close"]
    price_data = price_data.dropna()

    if not price_data.empty:
        # Step 2: Normalize prices and calculate portfolio value
        normalized_prices = price_data[tickers] / price_data[tickers].iloc[0]
        allocations = normalized_prices.mul([weights[t] for t in tickers], axis=1).mul(initial_capital)
        portfolio_value = allocations.sum(axis=1)

        # Step 3: Normalize S&P 500 for comparison
        sp500 = price_data["^GSPC"]
        sp500_norm = sp500 / sp500.iloc[0] * 100
        portfolio_norm = portfolio_value / portfolio_value.iloc[0] * 100

        # Step 4: Compute performance metrics
        returns = portfolio_value.pct_change().dropna()
        cumulative_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(portfolio_value)) - 1
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Prepare results
        metrics = {
            "Cumulative Return": f"{cumulative_return:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }
        print(f"Métricas del Backtest: {metrics}")

        # Create comparison plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_norm.index, y=portfolio_norm.values,
            mode='lines', name='Sentiment Portfolio',
            line=dict(color='royalblue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=sp500_norm.index, y=sp500_norm.values,
            mode='lines', name='S&P 500 Index',
            line=dict(color='firebrick', width=2, dash='dot')
        ))
        fig.update_layout(
            title='Sentiment Strategy vs S&P 500 (2024)',
            xaxis_title='Date',
            yaxis_title='Indexed Value (Start = 100)',
            template='plotly_white',
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        print("Mostrando gráfico de backtest en el navegador...")
        fig.show()
    else:
        print("No se pudieron descargar datos de precios para el backtest.")
else:
    print("No hay tickers válidos para ejecutar el backtest.")


# --- [CELDA 10: Fin] ---
print("\n--- [FIN DEL SCRIPT] ---")