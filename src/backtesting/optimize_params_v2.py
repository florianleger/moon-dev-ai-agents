#!/usr/bin/env python3
"""
Fine-grained parameter optimization around the best region found:
sl_mult=3.0, threshold=55, min_conv=2.
Also tests MIN_RR_RATIO variations.
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import src.backtesting.backtest_adaptive_hybrid as bt

print("Fetching BTC 1h 180d data...")
df_btc = bt.fetch_ohlcv_hyperliquid('BTC', '1h', 180)
print(f"Loaded {len(df_btc)} candles")

# Fine grid around best region
SL_MULTS = [2.5, 2.8, 3.0, 3.2, 3.5]
TP_MULTS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
BASE_THRESHOLDS = [50, 52, 55, 58, 60]
MIN_RR_RATIOS = [1.5, 1.8, 2.0]
# Fix min_conv=2 (best from v1)
MIN_CONV = 2

results = []
total_combos = len(SL_MULTS) * len(TP_MULTS) * len(BASE_THRESHOLDS) * len(MIN_RR_RATIOS)
i = 0

for sl, tp, thresh, rr in itertools.product(SL_MULTS, TP_MULTS, BASE_THRESHOLDS, MIN_RR_RATIOS):
    i += 1
    # Skip impossible combinations
    if tp < sl * 1.0:
        continue

    bt.ATR_PROFILES['major']['sl_mult'] = sl
    bt.ATR_PROFILES['major']['tp_mult'] = tp
    bt.BASE_THRESHOLD = thresh
    bt.MIN_CONVERGENT_MODULES = MIN_CONV
    bt.MIN_RR_RATIO = rr

    try:
        result = bt.run_backtest(df_btc.copy(), 'BTC', 500.0, verbose=False)
    except Exception as e:
        continue

    sl_exits = sum(1 for t in result.trades if t.exit_reason == 'SL')
    tp_exits = sum(1 for t in result.trades if t.exit_reason == 'TP')
    avg_mae = np.mean([t.max_adverse for t in result.trades]) * 100 if result.trades else 0
    avg_mfe = np.mean([t.max_favorable for t in result.trades]) * 100 if result.trades else 0

    row = {
        'sl_mult': sl, 'tp_mult': tp, 'threshold': thresh, 'min_rr': rr,
        'return_pct': result.total_return_pct,
        'sharpe': result.sharpe_ratio,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'max_dd': result.max_drawdown_pct,
        'trades': result.total_trades,
        'sl_exits': sl_exits, 'tp_exits': tp_exits,
        'avg_mae': avg_mae, 'avg_mfe': avg_mfe,
        'alpha': result.alpha,
    }
    results.append(row)

    flag = " ***" if result.profit_factor >= 1.0 else ""
    if i % 20 == 0 or result.profit_factor >= 1.0:
        print(f"  [{i}/{total_combos}] sl={sl:.1f} tp={tp:.1f} thresh={thresh} rr={rr} -> "
              f"ret={result.total_return_pct:+.1f}% WR={result.win_rate:.0f}% "
              f"PF={result.profit_factor:.2f} trades={result.total_trades} "
              f"DD={result.max_drawdown_pct:.1f}% SL/TP={sl_exits}/{tp_exits}{flag}")

# Sort and display
if results:
    df_r = pd.DataFrame(results)
    # Composite: emphasize profit factor and sharpe, penalize drawdown
    df_r['composite'] = (
        df_r['profit_factor'].clip(0, 3) / 3 * 40 +
        df_r['sharpe'].clip(-5, 5) / 5 * 30 +
        (1 - df_r['max_dd'] / 100) * 30
    )
    df_r = df_r.sort_values('composite', ascending=False)

    print("\n" + "=" * 120)
    print("TOP 15 PARAMETER COMBINATIONS (fine grid)")
    print("=" * 120)
    cols = ['sl_mult', 'tp_mult', 'threshold', 'min_rr', 'return_pct', 'sharpe',
            'win_rate', 'profit_factor', 'max_dd', 'trades', 'sl_exits', 'tp_exits',
            'avg_mae', 'avg_mfe', 'composite']
    print(df_r[cols].head(15).to_string(index=False))

    out_path = os.path.join(os.path.dirname(__file__), 'optimization_results_v2.csv')
    df_r.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Also show profitable combos
    profitable = df_r[df_r['profit_factor'] >= 1.0]
    print(f"\n{len(profitable)} profitable combinations found out of {len(df_r)} tested")
