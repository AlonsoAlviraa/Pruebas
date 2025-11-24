#!/usr/bin/env python3
"""
ANÁLISIS DE TRADES - Por qué tantos trades?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ejecutar el backtest consolidado y analizar
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_consolidated import run_consolidated_backtest

def analyze_trades(trades_df, config_name):
    """Análisis profundo de los trades"""
    
    print(f"\n{'='*70}")
    print(f"  ANÁLISIS DETALLADO - {config_name}")
    print(f"{'='*70}")
    
    if trades_df.empty:
        print("No hay trades para analizar")
        return
    
    # 1. ESTADÍSTICAS BÁSICAS
    print(f"\n[1] ESTADÍSTICAS GENERALES:")
    print(f"  Total trades: {len(trades_df)}")
    print(f"  Trades ganadores: {(trades_df['profit'] > 0).sum()} ({(trades_df['profit'] > 0).mean():.2%})")
    print(f"  Trades perdedores: {(trades_df['profit'] <= 0).sum()} ({(trades_df['profit'] <= 0).mean():.2%})")
    
    # 2. DURACIÓN DE TRADES (HOLDING PERIOD)
    trades_df['holding_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
    
    print(f"\n[2] HOLDING PERIOD (días):")
    print(f"  Promedio: {trades_df['holding_days'].mean():.1f} días")
    print(f"  Mediana: {trades_df['holding_days'].median():.0f} días")
    print(f"  Mín: {trades_df['holding_days'].min():.0f} días")
    print(f"  Máx: {trades_df['holding_days'].max():.0f} días")
    
    # Distribución de holding period
    holding_dist = {
        '1 día (day trading)': (trades_df['holding_days'] == 1).sum(),
        '2-5 días (swing corto)': ((trades_df['holding_days'] >= 2) & (trades_df['holding_days'] <= 5)).sum(),
        '6-20 días (swing medio)': ((trades_df['holding_days'] >= 6) & (trades_df['holding_days'] <= 20)).sum(),
        '21-60 días (swing largo)': ((trades_df['holding_days'] >= 21) & (trades_df['holding_days'] <= 60)).sum(),
        '>60 días (posición)': (trades_df['holding_days'] > 60).sum()
    }
    
    print(f"\n  Distribución de duración:")
    for period, count in holding_dist.items():
        pct = count / len(trades_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {period:<25} {count:>4} ({pct:>5.1f}%) {bar}")
    
    # 3. TRADES MUY CORTOS (problema potencial)
    very_short = trades_df[trades_df['holding_days'] <= 3]
    if len(very_short) > 0:
        print(f"\n[3] TRADES MUY CORTOS (≤3 días): {len(very_short)} ({len(very_short)/len(trades_df):.2%})")
        print(f"  Profit promedio: ${very_short['profit'].mean():.2f}")
        print(f"  Win rate: {(very_short['profit'] > 0).mean():.2%}")
        print(f"  ⚠️  PROBLEMA: Muchos trades cortos → Comisiones altas")
    
    # 4. DISTRIBUCIÓN POR TICKER
    trades_per_ticker = trades_df.groupby('ticker').size().sort_values(ascending=False)
    
    print(f"\n[4] DISTRIBUCIÓN POR TICKER:")
    print(f"  Tickers únicos operados: {len(trades_per_ticker)}")
    print(f"  Trades por ticker (promedio): {trades_per_ticker.mean():.1f}")
    
    print(f"\n  Top 10 tickers más operados:")
    for ticker, count in trades_per_ticker.head(10).items():
        pct = count / len(trades_df) * 100
        print(f"    {ticker:<6} {count:>4} trades ({pct:>5.1f}%)")
    
    # Detectar si hay tickers "sobre-operados"
    overtraded = trades_per_ticker[trades_per_ticker > 20]
    if len(overtraded) > 0:
        print(f"\n  ⚠️  ADVERTENCIA: {len(overtraded)} tickers con >20 trades:")
        print(f"      Esto sugiere REBALANCEO FRECUENTE en los mismos tickers")
    
    # 5. RENTABILIDAD
    print(f"\n[5] RENTABILIDAD:")
    print(f"  Profit total: ${trades_df['profit'].sum():,.2f}")
    print(f"  Profit promedio: ${trades_df['profit'].mean():.2f}")
    print(f"  Retorno promedio: {trades_df['return'].mean():.2%}")
    
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    
    if len(winners) > 0:
        print(f"\n  Trades ganadores:")
        print(f"    Profit promedio: ${winners['profit'].mean():.2f}")
        print(f"    Retorno promedio: {winners['return'].mean():.2%}")
    
    if len(losers) > 0:
        print(f"\n  Trades perdedores:")
        print(f"    Pérdida promedio: ${losers['profit'].mean():.2f}")
        print(f"    Retorno promedio: {losers['return'].mean():.2%}")
    
    # 6. RATIO RISK/REWARD
    if len(winners) > 0 and len(losers) > 0:
        avg_win = winners['profit'].mean()
        avg_loss = abs(losers['profit'].mean())
        ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"\n[6] RISK/REWARD:")
        print(f"  Ratio: {ratio:.2f}:1")
        if ratio < 1.5:
            print(f"  ⚠️  BAJO: Necesitas >60% win rate para ser rentable")
        elif ratio > 2.0:
            print(f"  ✓ BUENO: Rentable con win rate >40%")
    
    # 7. PROBLEMAS DETECTADOS Y SOLUCIONES
    print(f"\n{'='*70}")
    print(f"  DIAGNÓSTICO Y RECOMENDACIONES")
    print(f"{'='*70}")
    
    issues = []
    recommendations = []
    
    # Problema 1: Muchos trades cortos
    short_pct = len(very_short) / len(trades_df)
    if short_pct > 0.30:
        issues.append(f"❌ {short_pct:.0%} de trades duran ≤3 días (demasiado cortos)")
        recommendations.append("→ Agregar filtro: holding_period_min = 5 días")
    
    # Problema 2: Demasiados trades en general
    trades_per_year = len(trades_df) / 2  # 2 años de datos
    if trades_per_year > 500:
        issues.append(f"❌ {trades_per_year:.0f} trades/año (muy activo)")
        recommendations.append("→ Aumentar min_confidence de {0.60} a {0.65-0.70}")
        recommendations.append("→ Reducir max_positions de {10} a {5-8}")
    
    # Problema 3: Sobre-trading en pocos tickers
    if len(overtraded) > 0:
        issues.append(f"❌ {len(overtraded)} tickers sobre-operados (>20 trades)")
        recommendations.append("→ Agregar filtro: max_trades_per_ticker = 10")
        recommendations.append("→ Aumentar cooldown entre trades del mismo ticker")
    
    # Problema 4: Bajo risk/reward
    if len(winners) > 0 and len(losers) > 0:
        if ratio < 1.5:
            issues.append(f"❌ Risk/Reward bajo ({ratio:.2f}:1)")
            recommendations.append("→ Ajustar stops más amplios (aumentar k_atr)")
            recommendations.append("→ Dejar correr ganadores más tiempo")
    
    if issues:
        print(f"\n❌ PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"   {issue}")
        
        print(f"\n✓ SOLUCIONES RECOMENDADAS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print(f"\n✓ No se detectaron problemas mayores")
        print(f"  El sistema parece estar funcionando correctamente")
    
    return trades_df


def main():
    """Ejecutar análisis completo"""
    print("\n" + "="*70)
    print("  ANÁLISIS DE TRADES - ¿Por qué tantos?")
    print("="*70)
    
    # Cargar tickers
    ticker_file = Path("good.txt")
    if ticker_file.exists():
        tickers = [line.strip().upper() for line in ticker_file.read_text().splitlines() if line.strip()]
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    # Ejecutar backtest CONSERVADOR (el que más trades tiene)
    print("\nEjecutando backtest CONSERVADOR para análisis...")
    print("-" * 70)
    
    results = run_consolidated_backtest(
        tickers=tickers,
        data_root=Path("data"),
        model_path=Path("models/trend_model_2015_2024_OPTUNA_FIXED.joblib"),
        start_date="2023-01-01",
        end_date="2024-12-31",
        min_confidence=0.60,  # CONSERVADOR
        init_cash=10000.0,
        fees=0.001,
        max_positions=10
    )
    
    if results is None or results['trades_df'].empty:
        print("ERROR: No se generaron trades")
        return
    
    # Analizar trades
    trades_df = analyze_trades(results['trades_df'], "CONSERVADOR")
    
    # Guardar trades para análisis adicional
    output_file = "trades_analysis.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\n📁 Trades guardados en: {output_file}")
    print(f"   Puedes abrirlo en Excel para análisis adicional")
    
    print("\n" + "="*70)
    print("  ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nTiempo total: {elapsed:.1f} segundos")
