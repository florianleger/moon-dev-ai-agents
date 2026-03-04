#!/usr/bin/env python3
"""Quick ETH-specific optimization."""

import sys, os, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import src.backtesting.backtest_adaptive_hybrid as bt

print("Fetching ETH 1h 180d...")
df_eth = bt.fetch_ohlcv_hyperliquid('ETH', '1h', 180)
print(f"Loaded {len(df_eth)} candles")

SL_MULTS = [3.0, 3.5, 4.0, 4.5, 5.0]
THRESHOLDS = [50, 55, 58, 60]
MIN_RR = [1.5, 1.8, 2.0]

results = []
for sl, thresh, rr in itertools.product(SL_MULTS, THRESHOLDS, MIN_RR):
    tp = sl * 2.0  # generous TP
    bt.ATR_PROFILES['major']['sl_mult'] = sl
    bt.ATR_PROFILES['major']['tp_mult'] = tp
    bt.BASE_THRESHOLD = thresh
    bt.MIN_CONVERGENT_MODULES = 2
    bt.MIN_RR_RATIO = rr

    result = bt.run_backtest(df_eth.copy(), 'ETH', 500.0, verbose=False)
    sl_exits = sum(1 for t in result.trades if t.exit_reason == 'SL')
    tp_exits = sum(1 for t in result.trades if t.exit_reason == 'TP')

    row = {'sl_mult': sl, 'tp_mult': tp, 'threshold': thresh, 'min_rr': rr,
           'return_pct': result.total_return_pct, 'sharpe': result.sharpe_ratio,
           'win_rate': result.win_rate, 'profit_factor': result.profit_factor,
           'max_dd': result.max_drawdown_pct, 'trades': result.total_trades,
           'sl_exits': sl_exits, 'tp_exits': tp_exits}
    results.append(row)

    flag = " ***" if result.profit_factor >= 1.0 else ""
    if result.profit_factor >= 0.9 or result.profit_factor >= 1.0:
        print(f"sl={sl:.1f} tp={tp:.1f} thresh={thresh} rr={rr} -> "
              f"ret={result.total_return_pct:+.1f}% WR={result.win_rate:.0f}% "
              f"PF={result.profit_factor:.2f} trades={result.total_trades} "
              f"DD={result.max_drawdown_pct:.1f}% SL/TP={sl_exits}/{tp_exits}{flag}")

df_r = pd.DataFrame(results)
df_r['composite'] = (df_r['profit_factor'].clip(0,3)/3*40 + df_r['sharpe'].clip(-5,5)/5*30 + (1-df_r['max_dd']/100)*30)
df_r = df_r.sort_values('composite', ascending=False)
print("\nTOP 10 ETH COMBINATIONS:")
print(df_r.head(10).to_string(index=False))
