#!/usr/bin/env python3
"""
ANÁLISIS PROFUNDO - CONSERVADOR
Qué tickers funcionan, cuáles no, razones de salida, etc.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_trend_following import run_trend_following_backtest


def analyze_exit_reasons(trades_df):
    """Analiza las razones de salida"""
    print(f"\n{'='*70}")
    print(f"  [1] RAZONES DE SALIDA")
    print(f"{'='*70}")
    
    exit_counts = trades_df['exit_reason'].value_counts()
    
    print(f"\nDistribución de salidas:")
    for reason, count in exit_counts.items():
        pct = count / len(trades_df) * 100
        bar = "█" * int(pct / 2)
        
        reason_name = {
            'trailing_stop': 'Trailing Stop (protección)',
            'trend_broken': 'Tendencia Rota (close < MA50)',
            'end_of_period': 'Fin del Período (forzado)'
        }.get(reason, reason)
        
        print(f"  {reason_name:<35} {count:>3} ({pct:>5.1f}%) {bar}")
    
    # Comparar rentabilidad por tipo de salida
    print(f"\nRentabilidad por tipo de salida:")
    for reason in exit_counts.index:
        subset = trades_df[trades_df['exit_reason'] == reason]
        avg_profit = subset['profit'].mean()
        avg_return = subset['return'].mean()
        win_rate = (subset['profit'] > 0).mean()
        
        reason_name = {
            'trailing_stop': 'Trailing Stop',
            'trend_broken': 'Tendencia Rota',
            'end_of_period': 'Fin Período'
        }.get(reason, reason)
        
        print(f"  {reason_name:<20} Profit: ${avg_profit:>7.2f} | Return: {avg_return:>6.2%} | Win: {win_rate:>5.1%}")


def analyze_by_ticker(trades_df):
    """Analiza performance por ticker"""
    print(f"\n{'='*70}")
    print(f"  [2] ANÁLISIS POR TICKER")
    print(f"{'='*70}")
    
    # Agrupar por ticker
    ticker_stats = trades_df.groupby('ticker').agg({
        'profit': ['sum', 'mean', 'count'],
        'return': 'mean',
        'holding_days': 'mean'
    }).round(2)
    
    ticker_stats.columns = ['Total_Profit', 'Avg_Profit', 'Trades', 'Avg_Return', 'Avg_Hold_Days']
    ticker_stats['Win_Rate'] = trades_df.groupby('ticker')['profit'].apply(lambda x: (x > 0).mean())
    ticker_stats = ticker_stats.sort_values('Total_Profit', ascending=False)
    
    print(f"\n[TOP 15 MEJORES TICKERS]")
    print(f"{'Ticker':<8} {'Trades':>6} {'Total $':>10} {'Avg $':>8} {'Avg Ret':>8} {'Win%':>6} {'AvgHold':>8}")
    print("-" * 70)
    
    for ticker, row in ticker_stats.head(15).iterrows():
        print(f"{ticker:<8} {row['Trades']:>6.0f} ${row['Total_Profit']:>9,.2f} "
              f"${row['Avg_Profit']:>7.2f} {row['Avg_Return']:>7.2%} "
              f"{row['Win_Rate']:>5.1%} {row['Avg_Hold_Days']:>7.1f}d")
    
    print(f"\n[TOP 10 PEORES TICKERS]")
    print(f"{'Ticker':<8} {'Trades':>6} {'Total $':>10} {'Avg $':>8} {'Avg Ret':>8} {'Win%':>6} {'AvgHold':>8}")
    print("-" * 70)
    
    for ticker, row in ticker_stats.tail(10).iterrows():
        print(f"{ticker:<8} {row['Trades']:>6.0f} ${row['Total_Profit']:>9,.2f} "
              f"${row['Avg_Profit']:>7.2f} {row['Avg_Return']:>7.2%} "
              f"{row['Win_Rate']:>5.1%} {row['Avg_Hold_Days']:>7.1f}d")
    
    # Estadísticas generales
    print(f"\n[ESTADÍSTICAS GENERALES]")
    print(f"  Tickers únicos operados: {len(ticker_stats)}")
    print(f"  Tickers rentables: {(ticker_stats['Total_Profit'] > 0).sum()} ({(ticker_stats['Total_Profit'] > 0).mean():.1%})")
    print(f"  Tickers no rentables: {(ticker_stats['Total_Profit'] <= 0).sum()} ({(ticker_stats['Total_Profit'] <= 0).mean():.1%})")
    
    return ticker_stats


def analyze_winning_vs_losing(trades_df):
    """Analiza trades ganadores vs perdedores"""
    print(f"\n{'='*70}")
    print(f"  [3] GANADORES vs PERDEDORES")
    print(f"{'='*70}")
    
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    
    print(f"\n[TRADES GANADORES] ({len(winners)} trades, {len(winners)/len(trades_df):.1%})")
    print(f"  Profit promedio:    ${winners['profit'].mean():,.2f}")
    print(f"  Return promedio:    {winners['return'].mean():.2%}")
    print(f"  Holding promedio:   {winners['holding_days'].mean():.1f} días")
    print(f"  Mejor trade:        ${winners['profit'].max():,.2f} ({winners['return'].max():.2%})")
    
    print(f"\n[TRADES PERDEDORES] ({len(losers)} trades, {len(losers)/len(trades_df):.1%})")
    print(f"  Pérdida promedio:   ${losers['profit'].mean():,.2f}")
    print(f"  Return promedio:    {losers['return'].mean():.2%}")
    print(f"  Holding promedio:   {losers['holding_days'].mean():.1f} días")
    print(f"  Peor trade:         ${losers['profit'].min():,.2f} ({losers['return'].min():.2%})")
    
    # Razones de salida por ganadores/perdedores
    print(f"\n[SALIDAS - GANADORES]")
    winner_exits = winners['exit_reason'].value_counts()
    for reason, count in winner_exits.items():
        pct = count / len(winners) * 100
        reason_name = {
            'trailing_stop': 'Trailing Stop',
            'trend_broken': 'Tendencia Rota',
            'end_of_period': 'Fin Período'
        }.get(reason, reason)
        print(f"  {reason_name:<20} {count:>3} ({pct:>5.1f}%)")
    
    print(f"\n[SALIDAS - PERDEDORES]")
    loser_exits = losers['exit_reason'].value_counts()
    for reason, count in loser_exits.items():
        pct = count / len(losers) * 100
        reason_name = {
            'trailing_stop': 'Trailing Stop',
            'trend_broken': 'Tendencia Rota',
            'end_of_period': 'Fin Período'
        }.get(reason, reason)
        print(f"  {reason_name:<20} {count:>3} ({pct:>5.1f}%)")


def analyze_holding_periods(trades_df):
    """Analiza distribución de holding periods"""
    print(f"\n{'='*70}")
    print(f"  [4] DISTRIBUCIÓN DE HOLDING PERIODS")
    print(f"{'='*70}")
    
    buckets = [
        (0, 7, 'Muy corto (0-7 días)'),
        (8, 14, 'Corto (8-14 días)'),
        (15, 30, 'Medio (15-30 días)'),
        (31, 60, 'Largo (31-60 días)'),
        (61, 999, 'Muy largo (60+ días)')
    ]
    
    print(f"\nDistribución:")
    for min_d, max_d, label in buckets:
        subset = trades_df[(trades_df['holding_days'] >= min_d) & (trades_df['holding_days'] <= max_d)]
        count = len(subset)
        pct = count / len(trades_df) * 100
        
        if count > 0:
            avg_profit = subset['profit'].mean()
            win_rate = (subset['profit'] > 0).mean()
            bar = "█" * int(pct / 2)
            
            print(f"  {label:<25} {count:>3} ({pct:>5.1f}%) {bar}")
            print(f"    → Profit: ${avg_profit:>7.2f} | Win Rate: {win_rate:>5.1%}")


def find_patterns(trades_df):
    """Encuentra patrones en los mejores trades"""
    print(f"\n{'='*70}")
    print(f"  [5] PATRONES EN LOS MEJORES TRADES")
    print(f"{'='*70}")
    
    # Top 20 mejores trades
    top_trades = trades_df.nlargest(20, 'profit')
    
    print(f"\n[TOP 20 MEJORES TRADES]")
    print(f"{'Ticker':<8} {'Entry':>12} {'Exit':>12} {'Days':>5} {'Profit':>10} {'Return':>8} {'Reason':<15}")
    print("-" * 80)
    
    for _, trade in top_trades.iterrows():
        reason_short = {
            'trailing_stop': 'Stop',
            'trend_broken': 'Trend-Break',
            'end_of_period': 'End'
        }.get(trade['exit_reason'], trade['exit_reason'])
        
        print(f"{trade['ticker']:<8} {trade['entry_date'].strftime('%Y-%m-%d')} "
              f"{trade['exit_date'].strftime('%Y-%m-%d')} {trade['holding_days']:>5.0f} "
              f"${trade['profit']:>9,.2f} {trade['return']:>7.2%} {reason_short:<15}")
    
    # Patrones comunes
    print(f"\n[PATRONES COMUNES EN TOP TRADES]")
    print(f"  Holding promedio:     {top_trades['holding_days'].mean():.1f} días")
    print(f"  Retorno promedio:     {top_trades['return'].mean():.2%}")
    
    reason_dist = top_trades['exit_reason'].value_counts()
    print(f"\n  Salidas más comunes:")
    for reason, count in reason_dist.items():
        reason_name = {
            'trailing_stop': 'Trailing Stop',
            'trend_broken': 'Tendencia Rota',
            'end_of_period': 'Fin Período'
        }.get(reason, reason)
        print(f"    {reason_name:<20} {count}/20 ({count/20*100:.0f}%)")


def recommendations(trades_df, ticker_stats):
    """Genera recomendaciones basadas en análisis"""
    print(f"\n{'='*70}")
    print(f"  [6] RECOMENDACIONES")
    print(f"{'='*70}")
    
    # Tickers a evitar
    bad_tickers = ticker_stats[ticker_stats['Total_Profit'] < -100].index.tolist()
    if bad_tickers:
        print(f"\n❌ TICKERS A EXCLUIR (pérdida > $100):")
        for ticker in bad_tickers[:10]:
            profit = ticker_stats.loc[ticker, 'Total_Profit']
            trades = ticker_stats.loc[ticker, 'Trades']
            print(f"  {ticker}: ${profit:.2f} en {trades:.0f} trades")
    
    # Tickers ganadores consistentes
    good_tickers = ticker_stats[
        (ticker_stats['Total_Profit'] > 100) & 
        (ticker_stats['Win_Rate'] > 0.55) &
        (ticker_stats['Trades'] >= 3)
    ].index.tolist()
    
    if good_tickers:
        print(f"\n✓ TICKERS DE ALTO RENDIMIENTO:")
        for ticker in good_tickers[:10]:
            profit = ticker_stats.loc[ticker, 'Total_Profit']
            win_rate = ticker_stats.loc[ticker, 'Win_Rate']
            trades = ticker_stats.loc[ticker, 'Trades']
            print(f"  {ticker}: ${profit:.2f} | Win: {win_rate:.1%} | {trades:.0f} trades")
    
    # Análisis de salidas
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    
    stop_winners = (winners['exit_reason'] == 'trailing_stop').sum()
    trend_winners = (winners['exit_reason'] == 'trend_broken').sum()
    
    print(f"\n✓ EFECTIVIDAD DE SALIDAS:")
    print(f"  Ganadores por Trailing Stop: {stop_winners}/{len(winners)} ({stop_winners/len(winners)*100:.1f}%)")
    print(f"  Ganadores por Tendencia Rota: {trend_winners}/{len(winners)} ({trend_winners/len(winners)*100:.1f}%)")
    
    if stop_winners > trend_winners:
        print(f"\n  → El trailing stop está funcionando MUY BIEN")
        print(f"  → Protege ganancias efectivamente")
    else:
        print(f"\n  → Sales por tendencia rota son más comunes en ganadores")
        print(f"  → Considera ajustar k_atr para stops más amplios")


def main():
    """Ejecutar análisis completo"""
    print("\n" + "="*70)
    print("  ANÁLISIS PROFUNDO - CONSERVADOR")
    print("="*70)
    
    # Cargar tickers
    ticker_file = Path("good.txt")
    if ticker_file.exists():
        tickers = [line.strip().upper() for line in ticker_file.read_text().splitlines() if line.strip()]
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    # Ejecutar backtest CONSERVADOR
    print("\nEjecutando backtest CONSERVADOR...")
    
    results = run_trend_following_backtest(
        tickers=tickers,
        data_root=Path("data"),
        model_path=Path("models/trend_model_2015_2024_OPTUNA_FIXED.joblib"),
        start_date="2023-01-01",
        end_date="2024-12-31",
        min_confidence=0.65,
        init_cash=10000.0,
        fees=0.001,
        max_positions=10,
        k_atr=2.5,
        holding_period_min=15,
        cooldown_days=10
    )
    
    if not results or results['trades_df'].empty:
        print("ERROR: No se generaron trades")
        return
    
    trades_df = results['trades_df']
    
    # Análisis
    analyze_exit_reasons(trades_df)
    ticker_stats = analyze_by_ticker(trades_df)
    analyze_winning_vs_losing(trades_df)
    analyze_holding_periods(trades_df)
    find_patterns(trades_df)
    recommendations(trades_df, ticker_stats)
    
    # Guardar
    output_file = "analysis_conservador.csv"
    trades_df.to_csv(output_file, index=False)
    
    ticker_file = "analysis_by_ticker.csv"
    ticker_stats.to_csv(ticker_file)
    
    print(f"\n{'='*70}")
    print(f"  ANÁLISIS COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📁 Archivos guardados:")
    print(f"  - {output_file} (todos los trades)")
    print(f"  - {ticker_file} (estadísticas por ticker)")


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Tiempo: {elapsed:.1f} segundos")
