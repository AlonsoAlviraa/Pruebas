#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SCRIPT DE SCANNER DE ACCIONES v1.6
Autor: [Tu Nombre/Gemini]
Universidad: [Tu Universidad]
Fecha: 3/11/2025

Descripción:
Este script escanea el mercado de acciones de EE. UU. (usando un listado
completo del NASDAQ) en busca de compañías que cumplan 4 criterios:
1. Crecimiento Fundamental (EPS y Ingresos)
2. Momentum de Precio (3 meses)
3. Contracción de Volatilidad (1 semana)
4. Carácter de la Acción (ADR alto)

Uso:
1. Asegúrate de tener las librerías: pip install yfinance pandas tqdm
2. Coloca 'nasdaqlisted.txt' en la misma carpeta que este script.
3. Ejecuta el script: python stock_scanner.py
4. Los resultados se imprimirán y guardarán en 'scanner_resultados.csv'.

Historial de Cambios:
v1.6: Eliminado 'multiprocessing'. Se cambia a un enfoque serial (secuencial)
      para evitar todos los errores de "Rate Limit".
      Añadida la librería 'tqdm' para mostrar una barra de progreso,
      ya que el script ahora tardará mucho más en completarse.
v1.5: Ajustado CHUNK_SIZE (200->50) y COOLDOWN_SEC (10->15).
v1.4: Eliminado yf.set_tz_cache_location() para prevenir error 'database is locked'.
v1.3: Implementado procesamiento por lotes (chunking).
v1.2: Añadido df.dropna() para tickers.
"""

import yfinance as yf
import pandas as pd
import logging
import warnings
import time
from tqdm import tqdm # Importamos tqdm para la barra de progreso

# --- 1. CONFIGURACIÓN DE CRITERIOS ---
# Aquí puedes ajustar tus parámetros de filtrado.
# 1.1. Fundamentales (Crecimiento)
MIN_EPS_GROWTH = 0.50
MIN_REVENUE_GROWTH = 0.50

# 1.2. Momentum (Precio)
MIN_MOMENTUM_3M = 0.50

# 1.3. Contracción de Volatilidad (El Patrón)
MAX_VOLATILITY_1WK = 0.05

# 1.4. Carácter de la Acción
MIN_ADR = 0.05

# --- 2. CONFIGURACIÓN DEL SCRIPT ---

# [MOD v1.6] Pausa entre cada petición serial
# Pausa muy corta (en segundos) después de procesar CADA ticker.
# Esto es para ser "amigables" con la API de Yahoo Finance.
SERIAL_DELAY_SEC = 0.1

# Configura el logging para ver el progreso y los errores
# Subimos el nivel a WARNING para no ver los 'Rate Limit' si es que
# aun así ocurren (lo cual es poco probable).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suprime warnings de yfinance y pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def get_nasdaq_tickers(filename="nasdaqlisted.txt"):
    """
    Obtiene la lista de tickers del NASDAQ desde un archivo local (nasdaqlisted.txt).
    Filtra para incluir solo acciones comunes (excluye ETFs, Test Issues, etc.).
    """
    logging.info(f"Obteniendo listado de tickers del archivo '{filename}'...")
    try:
        df = pd.read_csv(filename, delimiter='|')

        # --- [FIX v1.2] LIMPIEZA DE DATOS ---
        df.dropna(subset=['Symbol', 'ETF', 'Test Issue'], inplace=True)

        # --- FILTROS DE CALIDAD ---
        df_filtered = df[df['ETF'] == 'N']
        df_filtered = df_filtered[df_filtered['Test Issue'] == 'N']
        df_filtered = df_filtered[df_filtered['Symbol'].str.contains(r'^[A-Z]+$')]

        tickers = df_filtered['Symbol'].tolist()
        
        logging.info(f"Se encontraron {len(tickers)} tickers de acciones comunes en '{filename}'.")
        return tickers
    
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo '{filename}'.")
        logging.error("Asegúrate de que 'nasdaqlisted.txt' esté en la misma carpeta que el script.")
        logging.warning("Usando una lista de respaldo corta (solo para demo).")
        return ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN']
    except Exception as e:
        logging.error(f"No se pudo procesar el archivo de tickers: {e}")
        logging.warning("Usando una lista de respaldo corta (solo para demo).")
        return ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN']

def check_single_stock(ticker_symbol):
    """
    Función que aplica todos los filtros a UN solo ticker.
    
    Devuelve un diccionario con datos si pasa los filtros, o None si falla.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Descarga de datos: .info para fundamentales, .history para precios
        info = ticker.info
        hist_3mo = ticker.history(period="3mo")
        hist_1wk = ticker.history(period="7d") # 7 días para asegurar 5 de trading

        # --- 1. CRITERIO: FUNDAMENTALES (CRECIMIENTO) ---
        eps_growth = info.get('earningsQuarterlyGrowth')
        rev_growth = info.get('revenueGrowth')

        if eps_growth is None or rev_growth is None:
            # logging.debug ya no es visible, usamos logging.info para fallos
            # logging.info(f"[{ticker_symbol}] DESCARTADO: Faltan datos fundamentales (EPS_G: {eps_growth}, REV_G: {rev_growth}).")
            return None
        
        if eps_growth < MIN_EPS_GROWTH or rev_growth < MIN_REVENUE_GROWTH:
            # logging.info(f"[{ticker_symbol}] DESCARTADO: Crecimiento insuficiente (EPS_G: {eps_growth:.2%}, REV_G: {rev_growth:.2%}).")
            return None

        # --- VALIDACIÓN DE DATOS HISTÓRICOS ---
        if hist_3mo.empty or len(hist_3mo) < 50 or hist_1wk.empty or len(hist_1wk) < 5:
            # logging.info(f"[{ticker_symbol}] DESCARTADO: Datos históricos insuficientes.")
            return None

        # --- 2. CRITERIO: MOMENTUM (PRECIO) ---
        price_now = hist_3mo['Close'].iloc[-1]
        price_3mo_ago = hist_3mo['Close'].iloc[0]

        if price_3mo_ago == 0: return None # Evitar división por cero
        
        momentum_3m = (price_now - price_3mo_ago) / price_3mo_ago

        if momentum_3m < MIN_MOMENTUM_3M:
            # logging.info(f"[{ticker_symbol}] DESCARTADO: Momentum 3M insuficiente ({momentum_3m:.2%}).")
            return None

        # --- 4. CRITERIO: CARÁCTER (ADR ALTO) ---
        hist_3mo['ADR_pct'] = (hist_3mo['High'] - hist_3mo['Low']) / hist_3mo['Close']
        adr_mean = hist_3mo['ADR_pct'].mean()

        if adr_mean < MIN_ADR:
            # logging.info(f"[{ticker_symbol}] DESCARTADO: ADR (carácter) insuficiente ({adr_mean:.2%}).")
            return None

        # --- 3. CRITERIO: CONTRACCIÓN DE VOLATILIDAD (1 SEMANA) ---
        hist_1wk = hist_1wk.iloc[-5:] 
        
        if len(hist_1wk) < 5: # Asegurarnos de tener 5 días
            return None

        wk_high = hist_1wk['High'].max()
        wk_low = hist_1wk['Low'].min()
        wk_ref_price = hist_1wk['Open'].iloc[0] # Precio referencia al inicio de la semana

        if wk_ref_price == 0: return None # Evitar división por cero

        volatility_1wk = (wk_high - wk_low) / wk_ref_price

        if volatility_1wk > MAX_VOLATILITY_1WK:
            # logging.info(f"[{ticker_symbol}] DESCARTADO: Contracción de volatilidad fallida. Vol={volatility_1wk:.2%}")
            return None

        # --- ¡ÉXITO! LA ACCIÓN CUMPLE TODO ---
        # (Ahora solo logueamos los ÉXITOS para no llenar la consola)
        logging.info(f"¡ÉXITO! {ticker_symbol} cumple todos los criterios.")
        
        result_data = {
            'Ticker': ticker_symbol,
            'Sector': info.get('sector', 'N/A'),
            'Industria': info.get('industry', 'N/A'),
            'Precio Actual': f"{price_now:.2f}",
            'Momentum 3M': f"{momentum_3m:.2%}",
            'ADR 3M (Carácter)': f"{adr_mean:.2%}",
            'Volatilidad 1WK': f"{volatility_1wk:.2%}",
            'Crec. EPS (Q)': f"{eps_growth:.2%}",
            'Crec. REV (Q)': f"{rev_growth:.2%}",
        }
        return result_data

    except Exception as e:
        # Captura cualquier error (datos faltantes, ticker deslistado, problemas de conexión)
        # Ocultamos errores '401' y '404' que son comunes por el rate limit o tickers malos
        if "401 Client Error" not in str(e) and "404 Client Error" not in str(e) and "No data found" not in str(e):
             logging.warning(f"Error procesando {ticker_symbol}: {e}")
        return None

def main():
    """
    Función principal del script.
    Orquesta la obtención de tickers, el análisis en serial y el guardado.
    """
    start_time = time.time()
    logging.info("--- Iniciando Scanner de Acciones (Estrategia Crecimiento + Contracción) ---")

    # 1. Obtener tickers
    tickers = get_nasdaq_tickers(filename="nasdaqlisted.txt")
    if not tickers:
        logging.error("No se obtuvieron tickers, finalizando script.")
        return
    
    # (Opcional) Limitar a N tickers para una prueba rápida
    # tickers = tickers[:200] 

    # --- [MOD v1.6] Procesamiento SERIAL ---
    # Se elimina el 'multiprocessing.Pool'
    # Se añade un bucle 'for' simple con 'tqdm' para la barra de progreso
    
    logging.info(f"Iniciando análisis de {len(tickers)} tickers en modo SERIAL (esto tardará)...")
    
    passed_stocks = []
    
    # tqdm envuelve la lista 'tickers' y muestra una barra de progreso
    for ticker_symbol in tqdm(tickers, desc="Analizando Tickers"):
        result = check_single_stock(ticker_symbol)
        if result:
            passed_stocks.append(result)
        
        # Pausa CORTA después de cada ticker para no saturar la API
        time.sleep(SERIAL_DELAY_SEC)


    # 3. Filtrar y guardar resultados
    end_time = time.time()
    logging.info(f"--- Análisis completado en {end_time - start_time:.2f} segundos ---")

    if not passed_stocks:
        logging.info("No se encontraron acciones que cumplan TODOS los criterios.")
        logging.warning("NOTA: El criterio de volatilidad < 1% en una semana (MAX_VOLATILITY_1WK) es extremadamente restrictivo.")
        logging.warning("Prueba a relajarlo (ej. 0.02 para 2%) si no obtienes resultados.")
    else:
        logging.info(f"¡Se encontraron {len(passed_stocks)} acciones que cumplen los criterios!")

        # Convertir lista de diccionarios a DataFrame de Pandas
        df_results = pd.DataFrame(passed_stocks)
        df_results.set_index('Ticker', inplace=True)

        # Guardar en CSV
        output_file = "scanner_resultados.csv"
        try:
            df_results.to_csv(output_file, encoding='utf-8-sig')
            logging.info(f"Resultados guardados en '{output_file}'")
        except Exception as e:
            logging.error(f"No se pudo guardar el archivo CSV: {e}")

        # Imprimir en consola
        print("\n" + "="*80)
        print("      ACCIONES QUE CUMPLEN LOS CRITERIOS")
        print("="*80)
        print(df_results.to_string())


if __name__ == "__main__":
    # Esta línea asegura que la función main() solo se ejecute
    # cuando el script es llamado directamente (no si es importado)
    main()

