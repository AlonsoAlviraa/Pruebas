#!/usr/bin/env python3
"""
Filtro de Calidad de Tickers (Quality Filter)
---------------------------------------------
Genera un universo de inversión limpio eliminando basura.
Criterios:
1. Precio Mínimo: > $0.30 (Evita Penny Stocks extremas y errores de split).
2. Liquidez Mínima: Volumen medio en dólares > $50,000/día.
3. Tipos de Activo: Elimina Warrants (W), Units (U), Rights (R) y Tests.
"""
import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("quality_filter")

def is_valid_ticker_symbol(ticker: str) -> bool:
    """Filtra por nomenclatura del símbolo (SPACs, Warrants, etc)."""
    ticker = ticker.upper().strip()
    
    # Eliminar Warrants (W), Rights (R), Units (U), Preferred (P) si están al final
    # A veces vienen como TICKERW o TICKER.W
    if len(ticker) > 4:
        # Patrones comunes de basura
        if ticker.endswith("W"): return False # Warrant
        if ticker.endswith("R"): return False # Right
        if ticker.endswith("U"): return False # Unit
        if ticker.endswith("P"): return False # Preferred
    
    # Eliminar tickers de prueba
    if "TEST" in ticker or "ZZZ" in ticker:
        return False
        
    return True

def check_data_quality(file_path: Path, min_price: float, min_dollar_volume: float) -> bool:
    """Abre el CSV y verifica precio y volumen recientes."""
    try:
        # Leer solo las últimas 50 filas para velocidad
        # Usamos engine='c' y on_bad_lines='skip' para robustez
        df = pd.read_csv(file_path)
        
        if df.empty or len(df) < 20:
            return False
            
        # Asegurar columnas numéricas
        cols = ["close", "volume"]
        # Normalizar nombres de columnas a minúsculas
        df.columns = df.columns.str.lower()
        
        if not all(c in df.columns for c in cols):
            return False
            
        # Tomar los últimos 20 días (aprox 1 mes de trading)
        recent = df.tail(20).copy()
        recent["close"] = pd.to_numeric(recent["close"], errors="coerce")
        recent["volume"] = pd.to_numeric(recent["volume"], errors="coerce")
        recent = recent.dropna()
        
        if recent.empty:
            return False
            
        # 1. Criterio de Precio
        avg_price = recent["close"].mean()
        if avg_price < min_price:
            return False
            
        # 2. Criterio de Liquidez (Dollar Volume)
        avg_dollar_vol = (recent["close"] * recent["volume"]).mean()
        if avg_dollar_vol < min_dollar_volume:
            return False
            
        return True
        
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Genera good_tickers.txt filtrando basura.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Carpeta de datos")
    parser.add_argument("--output", type=Path, default=Path("good_tickers.txt"), help="Archivo de salida")
    parser.add_argument("--min-price", type=float, default=0.30, help="Precio mínimo promedio")
    parser.add_argument("--min-liquidity", type=float, default=50000.0, help="Volumen diario en $ mínimo")
    
    args = parser.parse_args()
    
    if not args.data_root.exists():
        logger.error(f"Directorio {args.data_root} no encontrado.")
        return

    files = list(args.data_root.glob("*_history.csv"))
    logger.info(f"Analizando {len(files)} archivos en {args.data_root}...")
    
    valid_tickers = []
    rejected_counts = {"symbol": 0, "quality": 0}
    
    for f in tqdm(files, unit="ticker"):
        ticker = f.name.replace("_history.csv", "").strip().upper()
        
        # 1. Filtro de Símbolo (Rápido)
        if not is_valid_ticker_symbol(ticker):
            rejected_counts["symbol"] += 1
            continue
            
        # 2. Filtro de Datos (Lento - requiere leer archivo)
        if check_data_quality(f, args.min_price, args.min_liquidity):
            valid_tickers.append(ticker)
        else:
            rejected_counts["quality"] += 1
            
    # Guardar resultados
    valid_tickers.sort()
    with open(args.output, "w") as f:
        for t in valid_tickers:
            f.write(f"{t}\n")
            
    logger.info("-" * 40)
    logger.info(f"RESUMEN DEL FILTRO:")
    logger.info(f"Total Analizados:   {len(files)}")
    logger.info(f"Rechazados (Nombre): {rejected_counts['symbol']} (Warrants, Rights, etc)")
    logger.info(f"Rechazados (Datos):  {rejected_counts['quality']} (Penny stocks, ilíquidos)")
    logger.info(f"TICKERS VÁLIDOS:    {len(valid_tickers)}")
    logger.info(f"Guardado en:        {args.output}")
    logger.info("-" * 40)

if __name__ == "__main__":
    main()
