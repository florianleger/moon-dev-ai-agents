"""Order Imbalance scoring module (HyperLiquid L2 bid/ask depth)."""


def score_order_imbalance(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Order book bid/ask imbalance from HyperLiquid L2.

    Args:
        symbol: Token symbol.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        from hyperliquid.info import Info
        info = Info(skip_ws=True)
        l2_data = info.l2_snapshot(symbol)

        if not l2_data or 'levels' not in l2_data:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No L2 data'}

        levels = l2_data['levels']
        if len(levels) < 2:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient L2 data'}

        bids = levels[0]  # [[price, size], ...]
        asks = levels[1]

        # Sum top 10 levels
        bid_depth = sum(float(b['sz']) for b in bids[:10])
        ask_depth = sum(float(a['sz']) for a in asks[:10])
        total = bid_depth + ask_depth

        if total == 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Empty book'}

        imbalance = (bid_depth - ask_depth) / total  # -1 to +1

        long_score = 0
        short_score = 0

        if imbalance > 0.3:
            long_score += int(40 + imbalance * 30)  # 40-70
        elif imbalance < -0.3:
            short_score += int(40 + abs(imbalance) * 30)

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
        return {'score': min(100, best), 'direction': direction,
                'reason': f'Book imbalance={imbalance:+.2f} bid={bid_depth:.0f} ask={ask_depth:.0f}'}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'L2 error: {e}'}
