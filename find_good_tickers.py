#!/usr/bin/env python3
"""
Escanea el directorio de datos y lista los tickers que tienen
archivos de precios (history.csv) disponibles.
"""
import argparse
from pathlib import Path

# Nomenclatura de archivos (debe coincidir con tu downloader.py)
PRICE_PATTERN = "*_history.csv" 
PRICE_SUFFIX = "_history.csv"

def main():
    parser = argparse.ArgumentParser(
        description="Encuentra tickers con datos de precios válidos."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directorio raíz del caché de datos"
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"Error: El directorio {args.data_root} no existe.")
        return

    good_tickers = []
    for f in args.data_root.glob(PRICE_PATTERN):
        # Extraer el nombre del ticker del nombre del archivo
        # Ej: "AAL_history.csv" -> "AAL"
        ticker_name = f.name.replace(PRICE_SUFFIX, "")
        
        # --- CORRECCIÓN DE ROBUSTEZ ---
        # Aplicar .strip() para eliminar CUALQUIER espacio
        # (incluyendo \xa0) antes de la comprobación.
        ticker_name = ticker_name.strip()
        # ----------------------------

        if ticker_name: # Esto ahora filtrará tickers vacíos ("") o con espacios (" ")
            good_tickers.append(ticker_name)

    print(f"Encontrados {len(good_tickers)} tickers con datos de precios en {args.data_root}.")
    
    # Imprime la lista de tickers, uno por línea.
    # Puedes redirigir esto a un archivo:
    # python find_good_tickers.py > good_tickers.txt
    for ticker in sorted(good_tickers):
        print(ticker)

if __name__ == "__main__":
    main()