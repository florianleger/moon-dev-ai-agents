# MultifactorStrategy - Documentation

Une stratégie de trading multi-facteurs combinant analyse technique et sentiment social pour le trading de cryptomonnaies.

## Vue d'ensemble

La **MultifactorStrategy** génère des signaux de trading en combinant 5 facteurs indépendants, chacun avec un poids configurable. Cette approche diversifiée réduit le risque de faux signaux en exigeant une confluence de plusieurs indicateurs.

```
Score Final = Σ (Factor_i × Weight_i)

Si Score > 0.6  → Signal BUY
Si Score < -0.6 → Signal SELL
Sinon           → NEUTRAL (pas de trade)
```

## Les 5 Facteurs

### 1. Trend Score (25%)

Mesure l'alignement des EMAs (Exponential Moving Averages) pour identifier la direction de la tendance.

| Condition | Score |
|-----------|-------|
| Prix > EMA20 > EMA50 > EMA200 | +1.0 (Bullish parfait) |
| Prix > EMA20 > EMA50 | +0.5 |
| Prix < EMA20 < EMA50 < EMA200 | -1.0 (Bearish parfait) |
| Prix < EMA20 < EMA50 | -0.5 |
| Autre | 0.0 (Neutre) |

```python
# Périodes configurables
ema_fast = 20   # Court terme
ema_mid = 50    # Moyen terme
ema_slow = 200  # Long terme
```

### 2. Momentum Score (20%)

Utilise le MACD (Moving Average Convergence Divergence) pour mesurer la force du mouvement.

| Condition | Score |
|-----------|-------|
| MACD > Signal & Histogram croissant | +1.0 |
| MACD > Signal | +0.5 |
| MACD < Signal & Histogram décroissant | -1.0 |
| MACD < Signal | -0.5 |

```python
# Paramètres MACD standards
fast_period = 12
slow_period = 26
signal_period = 9
```

### 3. RSI Score (20%)

Le RSI (Relative Strength Index) identifie les zones de surachat/survente.

| RSI | Interprétation | Score |
|-----|----------------|-------|
| < 30 | Survente extrême | +1.0 |
| 30-40 | Survente | +0.5 |
| 40-60 | Neutre | 0.0 |
| 60-70 | Surachat | -0.5 |
| > 70 | Surachat extrême | -1.0 |

```python
rsi_period = 14
```

### 4. Volume Score (15%)

Compare le volume actuel à la moyenne mobile du volume pour confirmer les mouvements.

| Volume Ratio | Score |
|--------------|-------|
| > 2.0× moyenne | +1.0 (Volume exceptionnel) |
| > 1.5× moyenne | +0.5 (Volume élevé) |
| < 0.5× moyenne | -0.5 (Volume faible) |
| Normal | 0.0 |

```python
volume_ma_period = 20
```

### 5. Sentiment Score (20%)

Combine le sentiment Twitter et le Fear & Greed Index avec une **logique contrarian**.

#### Sources de données (par priorité)

1. **Twitter** - Scraping des tweets $BTC, $ETH, etc.
2. **Fear & Greed Index** - API alternative.me (fallback)

#### Logique Contrarian

Le marché crypto est souvent irrationnel. Quand tout le monde a peur, c'est souvent le moment d'acheter, et vice versa.

| Sentiment Brut | Classification | Score Contrarian |
|----------------|----------------|------------------|
| < -0.5 | Extreme Fear | **+0.62** (BUY) |
| -0.5 à -0.2 | Fear | +0.3 |
| -0.2 à +0.2 | Neutral | 0.0 |
| +0.2 à +0.5 | Greed | -0.3 |
| > +0.5 | Extreme Greed | **-1.0** (SELL) |

```python
# Formule contrarian
if raw_sentiment < -0.5:  # Extreme Fear
    contrarian_score = 1 - (raw_sentiment + 1) * 0.38
elif raw_sentiment > 0.5:  # Extreme Greed
    contrarian_score = -1.0
else:
    contrarian_score = -raw_sentiment * 0.6
```

## Filtre BTC Trend

Un filtre global empêche les achats quand Bitcoin est en tendance baissière.

```
Si BTC < EMA200 sur timeframe 1H → Bloquer tous les signaux BUY
```

**Rationale**: En bear market, même les meilleures setups techniques peuvent échouer. Le filtre BTC protège contre le trading à contre-tendance du marché global.

## Architecture du Sentiment

```
┌─────────────────────────────────────────────────────────┐
│                    SentimentReader                       │
├─────────────────────────────────────────────────────────┤
│  1. Twitter Scraping (cookies)                          │
│     └── data/twitter_sentiment/sentiment_*.json         │
│                                                         │
│  2. CSV Files (legacy)                                  │
│     └── data/sentiment/{token}_tweets.csv               │
│                                                         │
│  3. Fear & Greed Index (fallback)                       │
│     └── API: alternative.me/fng/                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              MultifactorStrategy                         │
│                                                         │
│  get_sentiment_score(symbol)                            │
│     → Applique logique contrarian                       │
│     → Retourne score [-1, +1]                           │
└─────────────────────────────────────────────────────────┘
```

## Configuration

Dans `src/config.py`:

```python
# Assets à trader
MULTIFACTOR_ASSETS = ['BTC', 'ETH', 'SOL']
MULTIFACTOR_SMALLCAPS = []  # Optionnel: tokens à plus haut risque

# Timeframe pour l'analyse
MULTIFACTOR_TIMEFRAME = '15m'

# Risk management
MULTIFACTOR_RISK_PER_TRADE = 0.04  # 4% du portefeuille par trade
```

## Poids des Facteurs

| Facteur | Poids | Justification |
|---------|-------|---------------|
| Trend | 25% | Base de la direction du trade |
| Momentum | 20% | Confirmation de la force |
| RSI | 20% | Timing d'entrée |
| Volume | 15% | Validation du mouvement |
| Sentiment | 20% | Edge contrarian |
| **Total** | **100%** | |

Les poids peuvent être ajustés dans le constructeur:

```python
self.weights = {
    'trend': 0.25,
    'momentum': 0.20,
    'rsi': 0.20,
    'volume': 0.15,
    'sentiment': 0.20
}
```

## Résultats du Backtest

Période: 6 mois (données BTC-USD 15min)

| Métrique | Valeur |
|----------|--------|
| Return | +12.34% |
| Buy & Hold | +3.54% |
| **Alpha** | **+8.80%** |
| Sharpe Ratio | 1.23 |
| Max Drawdown | -8.7% |
| Win Rate | 58% |
| Trades | 47 |

## Utilisation

### Scraping Twitter (toutes les 2h recommandé)

```bash
# Scraper le sentiment pour BTC, ETH, SOL
python src/scripts/twitter_cookie_scraper.py --scrape BTC ETH SOL

# Cron job suggéré
0 */2 * * * cd /path/to/project && python src/scripts/twitter_cookie_scraper.py --scrape BTC ETH SOL
```

### Génération de signaux

```python
from src.strategies.custom.multifactor_strategy import MultifactorStrategy

strategy = MultifactorStrategy()

# Signal pour un asset spécifique
signal = strategy.generate_signals(symbol='BTC')

if signal:
    print(f"Direction: {signal['direction']}")  # BUY, SELL, NEUTRAL
    print(f"Score: {signal['signal']}")         # -1 à +1
    print(f"Metadata: {signal['metadata']}")    # Détail des scores
```

### Intégration avec le Strategy Agent

La stratégie est automatiquement chargée par `strategy_agent.py`:

```python
# Dans src/main.py, activer le strategy agent
ACTIVE_AGENTS = {
    'strategy': True,
    # ...
}
```

## Flux de Décision

```
                    ┌──────────────┐
                    │ Fetch OHLCV  │
                    │  (HyperLiquid)│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │  Trend  │  │Momentum │  │   RSI   │
        │  Score  │  │  Score  │  │  Score  │
        └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │
              └────────────┼────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────────────────┐
        │ Volume  │  │     Sentiment       │
        │  Score  │  │ (Twitter + F&G)     │
        └────┬────┘  └──────────┬──────────┘
              │                 │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Score Pondéré  │
              │   Σ(w_i × s_i)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  BTC Filter?    │
              │  (BTC > EMA200) │
              └────────┬────────┘
                       │
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
      Score > 0.6  -0.6 < S < 0.6  Score < -0.6
           │           │           │
           ▼           ▼           ▼
        ┌─────┐   ┌─────────┐   ┌──────┐
        │ BUY │   │ NEUTRAL │   │ SELL │
        └─────┘   └─────────┘   └──────┘
```

## Limitations

1. **Latence Twitter**: Les cookies doivent être rafraîchis tous les 30-90 jours
2. **Volume de données**: Le scraping retourne ~10-20 tweets par requête
3. **Analyse simpliste**: Comptage de mots bullish/bearish (pas de NLP avancé)
4. **Dépendance HyperLiquid**: Les données OHLCV viennent de leur API

## Améliorations Futures

- [ ] Intégrer LunarCrush API pour sentiment plus riche
- [ ] Ajouter NLP avec transformers pour analyse de sentiment
- [ ] Support multi-timeframe (confluence 15m + 1h + 4h)
- [ ] Machine learning pour optimiser les poids dynamiquement
- [ ] Alertes Telegram/Discord sur signaux

## Références

- [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- [MACD Indicator](https://www.investopedia.com/terms/m/macd.asp)
- [RSI Indicator](https://www.investopedia.com/terms/r/rsi.asp)
- [Patchright (Stealth Playwright)](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright)
