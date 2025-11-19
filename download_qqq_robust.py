#!/usr/bin/env python3
"""
Script robusto para descargar QQQ usando el cliente de ANTIGUOPROGRAMA/a.py
que gestiona cookies y crumbs para evitar bloqueos.
"""
import sys
import logging
from pathlib import Path
import pandas as pd

# Añadir ANTIGUOPROGRAMA al path para poder importar a.py
sys.path.append(str(Path("ANTIGUOPROGRAMA").resolve()))

try:
    from a import YahooFinanceEUClient
except ImportError as e:
    print(f"Error importando YahooFinanceEUClient: {e}")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_qqq_robust")

def download_qqq_robust(output_dir: Path = Path("data")):
    ticker = "QQQ"
    output_path = output_dir / f"{ticker}_history.csv"
    
    logger.info(f"Iniciando descarga robusta de {ticker}...")
    
    try:
        client = YahooFinanceEUClient()
        
        # Usar el método fetch_history del cliente que ya maneja la sesión
        # Nota: fetch_history en a.py descarga 3 meses por defecto.
        # Vamos a usar el método interno _request para pedir más datos si es necesario,
        # o modificar fetch_history. Pero para no tocar a.py, usaremos _request
        # simulando lo que hace fetch_history pero con rango '5y' (5 años).
        
        params = {"range": "5y", "interval": "1d", "events": "history"}
        data = client._request(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params,
            expected_root="chart",
        )
        
        chart = data.get("chart", {})
        results = chart.get("result") or []
        if not results:
            raise ValueError("No hay datos de histórico")

        payload = results[0]
        timestamps = payload.get("timestamp") or []
        indicators = payload.get("indicators", {}).get("quote", [])
        
        if not timestamps or not indicators:
            raise ValueError("Datos incompletos en la respuesta")

        quotes = indicators[0]
        df = pd.DataFrame(quotes)
        df["date"] = pd.to_datetime(timestamps, unit="s", utc=True)
        
        # Seleccionar y renombrar columnas
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.dropna()
        df = df.set_index("date").sort_index()
        
        # Guardar
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        
        logger.info(f"Datos guardados en: {output_path}")
        logger.info(f"Rango: {df.index.min().date()} -> {df.index.max().date()}")
        logger.info(f"Filas: {len(df)}")
        
    except Exception as e:
        logger.error(f"Error en descarga robusta: {e}")
        # Fallback a yfinance simple si falla todo
        logger.info("Intentando fallback con yfinance simple...")
        try:
            import yfinance as yf
            df = yf.download(ticker, period="5y", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df.index.name = "date"
                df.to_csv(output_path)
                logger.info("Fallback exitoso.")
        except Exception as ex:
            logger.error(f"Fallback falló también: {ex}")

if __name__ == "__main__":
    download_qqq_robust()
