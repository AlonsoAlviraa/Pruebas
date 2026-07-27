#!/usr/bin/env python3
"""
TRIPLE BARRIER METHOD - López de Prado (legacy entrypoint)

SSOT implementation: trad_research.labels + LabelConfig (LAB-01).
This module keeps the historical CLI/API surface for older notebooks.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from trad_research.config import LabelConfig
from trad_research.labels import label_one_event

try:
    from drl_platform.data_pipeline import DataPipeline, PipelineConfig
except ImportError:  # optional for pure unit use
    DataPipeline = None  # type: ignore
    PipelineConfig = None  # type: ignore


def calculate_triple_barrier_label(
    data: pd.DataFrame,
    entry_idx: int,
    k_tp: float = 3.0,    # Take profit: 3× ATR (legacy default)
    k_sl: float = 2.0,    # Stop loss: 2× ATR
    max_hold: int = 20    # Máximo 20 días
) -> Tuple[int, int, float]:
    """
    Calcula label usando Triple Barrier Method (LdP encoding).

    Returns:
        label: 1 (BUY), -1 (SELL), 0 (HOLD)
        holding_days: Días hasta tocar barrera
        return_pct: Retorno al salir
    """
    entry_price = float(data.iloc[entry_idx]["close"])
    entry_atr = float(data.iloc[entry_idx]["atr"])
    if pd.isna(entry_atr) or entry_atr <= 0:
        return 0, 0, 0.0

    end = min(entry_idx + max_hold, len(data) - 1)
    if end <= entry_idx:
        return 0, 0, 0.0

    path = data.iloc[entry_idx + 1 : end + 1]
    cfg = LabelConfig(k_tp=k_tp, k_sl=k_sl, max_horizon=max_hold)
    label, days, ret = label_one_event(
        entry_price,
        path["high"].to_numpy(dtype=float),
        path["low"].to_numpy(dtype=float),
        entry_atr,
        config=cfg,
    )
    if label == 0 and days == max_hold and len(path) > 0:
        final_price = float(path.iloc[-1]["close"])
        ret = (final_price - entry_price) / entry_price
    return label, days, ret


def apply_triple_barrier_to_ticker(
    ticker: str,
    data_root: Path,
    k_tp: float = 3.0,
    k_sl: float = 2.0,
    max_hold: int = 20,
    sampling_freq: int = 5  # Etiquetar cada N días (no todos)
) -> pd.DataFrame:
    """
    Aplica Triple Barrier a un ticker completo
    """
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    
    # Cargar datos con indicadores
    df = pipeline.load_feature_view(ticker, indicators=True)
    
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    # Asegurar que tenemos ATR
    if 'atr' not in df.columns:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    df = df.dropna(subset=['atr']).reset_index(drop=True)
    
    if len(df) < 50:
        return pd.DataFrame()
    
    # Generar labels
    labels_data = []
    
    # Etiquetar cada N días (no todos los días para evitar overlap)
    for idx in range(0, len(df) - max_hold - 1, sampling_freq):
        label, holding_days, return_pct = calculate_triple_barrier_label(
            df, idx, k_tp, k_sl, max_hold
        )
        
        # Solo guardar si hay label válido
        if label != 0 or abs(return_pct) > 0.01:  # HOLD solo si tiene movimiento
            labels_data.append({
                'ticker': ticker,
                'date': df.iloc[idx]['date'],
                'label': label,
                'holding_days': holding_days,
                'return_pct': return_pct,
                'entry_price': df.iloc[idx]['close'],
                'atr': df.iloc[idx]['atr'],
                # Features para modelo (las mismas que antes)
                **{col: df.iloc[idx][col] for col in df.columns 
                   if col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']}
            })
    
    return pd.DataFrame(labels_data)


def generate_triple_barrier_dataset(
    ticker_file: Path,
    data_root: Path,
    output_file: Path,
    k_tp: float = 3.0,
    k_sl: float = 2.0,
    max_hold: int = 20,
    min_samples: int = 50,
    n_tickers: int = None  # Limitar para testing
):
    """
    Genera dataset completo con labels de Triple Barrier
    """
    print("="*70)
    print("  GENERANDO DATASET - TRIPLE BARRIER METHOD")
    print("="*70)
    print(f"\nParámetros:")
    print(f"  Take Profit: {k_tp}× ATR")
    print(f"  Stop Loss: {k_sl}× ATR")
    print(f"  Max Hold: {max_hold} días")
    
    # Cargar tickers
    if ticker_file.exists():
        tickers = [line.strip().upper() for line in ticker_file.read_text().splitlines() if line.strip()]
    else:
        print(f"ERROR: {ticker_file} no encontrado")
        return
    
    if n_tickers:
        tickers = tickers[:n_tickers]
    
    print(f"\nProcesando {len(tickers)} tickers...")
    
    all_labels = []
    
    # Procesar cada ticker
    for ticker in tqdm(tickers, desc="Generando labels"):
        try:
            ticker_labels = apply_triple_barrier_to_ticker(
                ticker, data_root, k_tp, k_sl, max_hold
            )
            
            if not ticker_labels.empty:
                all_labels.append(ticker_labels)
                
        except Exception as e:
            print(f"\nError en {ticker}: {e}")
            continue
    
    if not all_labels:
        print("\nERROR: No se generaron labels")
        return
    
    # Concatenar todo
    dataset = pd.concat(all_labels, ignore_index=True)
    
    # Filtrar tickers con muy pocos samples
    ticker_counts = dataset['ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= min_samples].index
    dataset = dataset[dataset['ticker'].isin(valid_tickers)]
    
    print(f"\n{'='*70}")
    print(f"  ESTADÍSTICAS DEL DATASET")
    print(f"{'='*70}")
    print(f"\nTotal samples: {len(dataset):,}")
    print(f"Tickers únicos: {dataset['ticker'].nunique()}")
    print(f"Rango fechas: {dataset['date'].min()} a {dataset['date'].max()}")
    
    print(f"\nDistribución de labels:")
    label_dist = dataset['label'].value_counts(normalize=True)
    for label, pct in label_dist.items():
        label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
        print(f"  {label_name:>6} ({label:>2}): {pct:>6.2%} ({dataset['label'].value_counts()[label]:>6,} samples)")
    
    print(f"\nHolding period:")
    print(f"  Promedio: {dataset['holding_days'].mean():.1f} días")
    print(f"  Mediana: {dataset['holding_days'].median():.0f} días")
    print(f"  Min/Max: {dataset['holding_days'].min():.0f} / {dataset['holding_days'].max():.0f} días")
    
    print(f"\nRetornos promedio por label:")
    for label in [1, -1, 0]:
        if label in dataset['label'].values:
            subset = dataset[dataset['label'] == label]
            avg_ret = subset['return_pct'].mean()
            label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
            print(f"  {label_name:>6}: {avg_ret:>7.2%}")
    
    # Guardar
    dataset.to_csv(output_file, index=False)
    print(f"\n✓ Dataset guardado en: {output_file}")
    
    return dataset


def analyze_triple_barrier_dataset(dataset_file: Path):
    """
    Analiza dataset generado con Triple Barrier
    """
    df = pd.read_csv(dataset_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print("="*70)
    print("  ANÁLISIS TRIPLE BARRIER DATASET")
    print("="*70)
    
    # 1. Balance de clases
    print(f"\n[1] BALANCE DE CLASES:")
    for label in [1, -1, 0]:
        count = (df['label'] == label).sum()
        pct = count / len(df)
        label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
        bar = "█" * int(pct * 50)
        print(f"  {label_name:>6}: {count:>6,} ({pct:>6.2%}) {bar}")
    
    # 2. Holding period por label
    print(f"\n[2] HOLDING PERIOD POR LABEL:")
    for label in [1, -1, 0]:
        subset = df[df['label'] == label]
        if len(subset) > 0:
            label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
            print(f"  {label_name:>6}: {subset['holding_days'].mean():>5.1f} días (median: {subset['holding_days'].median():>4.0f})")
    
    # 3. Retornos
    print(f"\n[3] RETORNOS POR LABEL:")
    for label in [1, -1, 0]:
        subset = df[df['label'] == label]
        if len(subset) > 0:
            label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
            print(f"  {label_name:>6}: {subset['return_pct'].mean():>7.2%} (std: {subset['return_pct'].std():>6.2%})")
    
    # 4. Top tickers
    print(f"\n[4] TOP 10 TICKERS MÁS REPRESENTADOS:")
    top_tickers = df['ticker'].value_counts().head(10)
    for ticker, count in top_tickers.items():
        pct = count / len(df)
        print(f"  {ticker:<6}: {count:>4} samples ({pct:>5.2%})")
    
    # 5. Distribución temporal
    print(f"\n[5] DISTRIBUCIÓN TEMPORAL:")
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year').size()
    for year, count in yearly.items():
        pct = count / len(df)
        bar = "█" * int(pct * 30)
        print(f"  {year}: {count:>5,} ({pct:>5.2%}) {bar}")


def main():
    """
    Ejecutar generación de dataset con Triple Barrier
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Triple Barrier Dataset")
    parser.add_argument("--ticker-file", type=Path, default=Path("good.txt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/triple_barrier_dataset.csv"))
    parser.add_argument("--k-tp", type=float, default=3.0, help="Take profit multiplier (ATR)")
    parser.add_argument("--k-sl", type=float, default=2.0, help="Stop loss multiplier (ATR)")
    parser.add_argument("--max-hold", type=int, default=20, help="Max holding days")
    parser.add_argument("--n-tickers", type=int, default=None, help="Limit tickers (for testing)")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing dataset")
    
    args = parser.parse_args()
    
    if args.analyze:
        if args.output.exists():
            analyze_triple_barrier_dataset(args.output)
        else:
            print(f"ERROR: {args.output} no existe")
    else:
        dataset = generate_triple_barrier_dataset(
            ticker_file=args.ticker_file,
            data_root=args.data_root,
            output_file=args.output,
            k_tp=args.k_tp,
            k_sl=args.k_sl,
            max_hold=args.max_hold,
            n_tickers=args.n_tickers
        )
        
        if dataset is not None:
            print(f"\n✓ Listo! Ahora puedes:")
            print(f"  1. Analizar: python triple_barrier_labeling.py --analyze")
            print(f"  2. Entrenar modelo: python train_signal_model_v2.py --dataset {args.output}")


if __name__ == "__main__":
    main()
