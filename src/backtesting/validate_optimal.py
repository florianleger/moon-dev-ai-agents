#!/usr/bin/env python3
"""
Validate optimal parameters with walk-forward and cross-asset testing.
Optimal params: sl_mult=2.8, tp_mult=3.5, threshold=55, min_rr=1.5, min_conv=2
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import src.backtesting.backtest_adaptive_hybrid as bt

# Set optimal parameters
bt.ATR_PROFILES['major']['sl_mult'] = 2.8
bt.ATR_PROFILES['major']['tp_mult'] = 3.5
bt.ATR_PROFILES['mid']['sl_mult'] = 2.2
bt.ATR_PROFILES['mid']['tp_mult'] = 2.8
bt.ATR_PROFILES['alt']['sl_mult'] = 1.8
bt.ATR_PROFILES['alt']['tp_mult'] = 2.3
bt.BASE_THRESHOLD = 55
bt.MIN_CONVERGENT_MODULES = 2
bt.MIN_RR_RATIO = 1.5

# -------------------------------------------------------------------------
# Test 1: BTC full backtest with optimal params
# -------------------------------------------------------------------------
print("=" * 70)
print("TEST 1: BTC 1h 180d - OPTIMAL PARAMS")
print("=" * 70)
print(f"Params: sl_mult=2.8, tp_mult=3.5, thresh=55, min_rr=1.5, min_conv=2")
print()

df_btc = bt.fetch_ohlcv_hyperliquid('BTC', '1h', 180)
print(f"Loaded {len(df_btc)} BTC candles")
result_btc = bt.run_backtest(df_btc, 'BTC', 500.0)

if result_btc.trades:
    sl_exits = sum(1 for t in result_btc.trades if t.exit_reason == 'SL')
    tp_exits = sum(1 for t in result_btc.trades if t.exit_reason == 'TP')
    avg_mae = np.mean([t.max_adverse for t in result_btc.trades]) * 100
    avg_mfe = np.mean([t.max_favorable for t in result_btc.trades]) * 100
    print(f"SL/TP ratio: {sl_exits}/{tp_exits}")
    print(f"MAE: {avg_mae:.2f}% | MFE: {avg_mfe:.2f}%")

# -------------------------------------------------------------------------
# Test 2: BTC Walk-Forward
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 2: BTC WALK-FORWARD ANALYSIS")
print("=" * 70)
bt.run_walk_forward(df_btc, 'BTC', train_days=90, test_days=30, initial_balance=500.0)

# -------------------------------------------------------------------------
# Test 3: ETH cross-asset validation
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 3: ETH 1h 180d - CROSS-ASSET VALIDATION")
print("=" * 70)

df_eth = bt.fetch_ohlcv_hyperliquid('ETH', '1h', 180)
print(f"Loaded {len(df_eth)} ETH candles")
result_eth = bt.run_backtest(df_eth, 'ETH', 500.0)

if result_eth.trades:
    sl_exits = sum(1 for t in result_eth.trades if t.exit_reason == 'SL')
    tp_exits = sum(1 for t in result_eth.trades if t.exit_reason == 'TP')
    avg_mae = np.mean([t.max_adverse for t in result_eth.trades]) * 100
    avg_mfe = np.mean([t.max_favorable for t in result_eth.trades]) * 100
    print(f"SL/TP ratio: {sl_exits}/{tp_exits}")
    print(f"MAE: {avg_mae:.2f}% | MFE: {avg_mfe:.2f}%")

# -------------------------------------------------------------------------
# Test 4: ETH Walk-Forward
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 4: ETH WALK-FORWARD ANALYSIS")
print("=" * 70)
bt.run_walk_forward(df_eth, 'ETH', train_days=90, test_days=30, initial_balance=500.0)

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)
print(f"BTC: ret={result_btc.total_return_pct:+.2f}% | WR={result_btc.win_rate:.1f}% | PF={result_btc.profit_factor:.2f} | DD={result_btc.max_drawdown_pct:.1f}% | Sharpe={result_btc.sharpe_ratio:.2f} | Trades={result_btc.total_trades}")
print(f"ETH: ret={result_eth.total_return_pct:+.2f}% | WR={result_eth.win_rate:.1f}% | PF={result_eth.profit_factor:.2f} | DD={result_eth.max_drawdown_pct:.1f}% | Sharpe={result_eth.sharpe_ratio:.2f} | Trades={result_eth.total_trades}")
