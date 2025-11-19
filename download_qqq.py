#!/usr/bin/env python3
"""
Script simple para descargar datos históricos de QQQ (Nasdaq 100 ETF)
y guardarlos en la carpeta data/ con el formato compatible.
"""
import logging
from pathlib import Path
import yfinance as yf
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_qqq")

def download_qqq_data(
    start_date: str = "2020-01-01",
    output_dir: Path = Path("data")
) -> None:
    
    ticker = "QQQ"
    output_path = output_dir / f"{ticker}_history.csv"
    
    logger.info(f"Iniciando descarga de {ticker} desde {start_date}...")
    
    try:
        # Descargar datos usando yfinance
        df = yf.download(ticker, start=start_date, progress=True)
        
        if df.empty:
            logger.error(f"No se descargaron datos para {ticker}")
            return

        # Aplanar MultiIndex si existe (común en versiones recientes de yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Renombrar columnas a minúsculas para compatibilidad con el pipeline
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close"
        })
        
        # Asegurar que el índice es datetime y tiene nombre 'date'
        df.index.name = "date"
        
        # Guardar a CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        
        logger.info(f"Datos guardados exitosamente en: {output_path}")
        logger.info(f"Total filas: {len(df)}")
        logger.info(f"Rango: {df.index.min().date()} -> {df.index.max().date()}")
        
    except Exception as e:
        logger.error(f"Error durante la descarga: {e}")

if __name__ == "__main__":
    download_qqq_data()
