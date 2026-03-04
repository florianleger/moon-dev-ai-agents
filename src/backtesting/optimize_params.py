#!/usr/bin/env python3
"""
Parameter optimization grid search for Adaptive Hybrid Strategy.
Tests multiple SL/TP multipliers, thresholds, and convergence settings.
"""

import sys
import os
import time
import itertools
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import src.backtesting.backtest_adaptive_hybrid as bt

# -------------------------------------------------------------------------
# Fetch data once
# -------------------------------------------------------------------------
print("Fetching BTC 1h 180d data...")
df_btc = bt.fetch_ohlcv_hyperliquid('BTC', '1h', 180)
print(f"Loaded {len(df_btc)} candles")

# -------------------------------------------------------------------------
# Parameter grid
# -------------------------------------------------------------------------
SL_MULTS = [2.0, 2.5, 3.0]
TP_MULTS = [3.0, 3.75, 4.5]
BASE_THRESHOLDS = [45, 50, 55]
MIN_CONVERGENT = [2, 3]

results = []
total = len(SL_MULTS) * len(TP_MULTS) * len(BASE_THRESHOLDS) * len(MIN_CONVERGENT)
i = 0

for sl, tp, thresh, conv in itertools.product(SL_MULTS, TP_MULTS, BASE_THRESHOLDS, MIN_CONVERGENT):
    i += 1
    # Skip if R:R < 1.2
    if tp / sl < 1.2:
        continue

    # Override global params
    bt.ATR_PROFILES['major']['sl_mult'] = sl
    bt.ATR_PROFILES['major']['tp_mult'] = tp
    bt.BASE_THRESHOLD = thresh
    bt.MIN_CONVERGENT_MODULES = conv

    try:
        result = bt.run_backtest(df_btc.copy(), 'BTC', 500.0, verbose=False)
    except Exception as e:
        print(f"  [{i}/{total}] sl={sl} tp={tp} thresh={thresh} conv={conv} -> ERROR: {e}")
        continue

    row = {
        'sl_mult': sl, 'tp_mult': tp, 'threshold': thresh, 'min_conv': conv,
        'return_pct': result.total_return_pct,
        'sharpe': result.sharpe_ratio,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'max_dd': result.max_drawdown_pct,
        'trades': result.total_trades,
        'avg_pnl': result.avg_trade_pnl,
        'alpha': result.alpha,
    }

    # Compute SL/TP exit breakdown
    sl_exits = sum(1 for t in result.trades if t.exit_reason == 'SL')
    tp_exits = sum(1 for t in result.trades if t.exit_reason == 'TP')
    avg_mae = np.mean([t.max_adverse for t in result.trades]) * 100 if result.trades else 0
    avg_mfe = np.mean([t.max_favorable for t in result.trades]) * 100 if result.trades else 0
    row['sl_exits'] = sl_exits
    row['tp_exits'] = tp_exits
    row['avg_mae'] = avg_mae
    row['avg_mfe'] = avg_mfe

    results.append(row)
    flag = " ***" if result.profit_factor >= 1.0 else ""
    print(f"  [{i}/{total}] sl={sl:.1f} tp={tp:.2f} thresh={thresh} conv={conv} -> "
          f"ret={result.total_return_pct:+.1f}% WR={result.win_rate:.0f}% "
          f"PF={result.profit_factor:.2f} trades={result.total_trades} "
          f"DD={result.max_drawdown_pct:.1f}% Sharpe={result.sharpe_ratio:.2f}{flag}")

# -------------------------------------------------------------------------
# Sort and display top results
# -------------------------------------------------------------------------
if results:
    df_results = pd.DataFrame(results)
    # Rank by composite: profit_factor * 0.4 + sharpe * 0.3 + (1 - max_dd/100) * 0.3
    df_results['composite'] = (
        df_results['profit_factor'].clip(0, 3) / 3 * 40 +
        df_results['sharpe'].clip(-5, 5) / 5 * 30 +
        (1 - df_results['max_dd'] / 100) * 30
    )
    df_results = df_results.sort_values('composite', ascending=False)

    print("\n" + "=" * 100)
    print("TOP 10 PARAMETER COMBINATIONS (by composite score)")
    print("=" * 100)
    cols = ['sl_mult', 'tp_mult', 'threshold', 'min_conv', 'return_pct', 'sharpe',
            'win_rate', 'profit_factor', 'max_dd', 'trades', 'sl_exits', 'tp_exits',
            'avg_mae', 'avg_mfe', 'composite']
    print(df_results[cols].head(10).to_string(index=False))

    # Save full results
    out_path = os.path.join(os.path.dirname(__file__), 'optimization_results.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")
