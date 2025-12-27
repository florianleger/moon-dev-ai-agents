"""
Sentiment Reader - Reads sentiment data from sentiment_agent outputs
Includes fallback to Fear & Greed Index when Twitter data unavailable
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from termcolor import cprint


class SentimentReader:
    """Reads and processes sentiment data from CSV files and Twitter scraping."""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent / "data"
        self.history_path = self.base_path / "sentiment_history.csv"
        self.sentiment_dir = self.base_path / "sentiment"
        self.twitter_dir = self.base_path / "twitter_sentiment"
        self._fear_greed_cache = None
        self._fear_greed_timestamp = None
        self._twitter_cache = {}
        self._twitter_cache_timestamp = None

    def get_current_sentiment(self, symbol: str) -> float:
        """
        Get the most recent sentiment score for a symbol.
        Priority: Twitter scraping > CSV files > Fear & Greed Index

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'bitcoin', 'ethereum')

        Returns:
            float: Sentiment score between -1 and +1, or 0.0 if no data
        """
        symbol_upper = symbol.upper()

        # 1. Try Twitter scraped sentiment first (most recent)
        twitter_sentiment = self._get_twitter_sentiment(symbol_upper)
        if twitter_sentiment is not None:
            return twitter_sentiment

        # Map common symbols to search terms for CSV files
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'xrp',
            'DOGE': 'dogecoin',
            'ADA': 'cardano',
            'AVAX': 'avalanche',
            'LINK': 'chainlink',
            'DOT': 'polkadot',
            'MATIC': 'polygon'
        }

        search_term = symbol_map.get(symbol_upper, symbol.lower())

        # 2. Try to read token-specific sentiment file
        token_file = self.sentiment_dir / f"{search_term}_tweets.csv"
        if token_file.exists():
            try:
                df = pd.read_csv(token_file)
                if not df.empty and 'sentiment_score' in df.columns:
                    recent = df.tail(10)
                    return float(recent['sentiment_score'].mean())
            except Exception:
                pass

        # 3. Fallback to global sentiment history (only if recent)
        if self.history_path.exists():
            try:
                df = pd.read_csv(self.history_path)
                if not df.empty and 'sentiment_score' in df.columns and 'timestamp' in df.columns:
                    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                    age = datetime.now() - last_timestamp
                    if age.total_seconds() < 7200:  # 2 hours
                        return float(df['sentiment_score'].iloc[-1])
            except Exception:
                pass

        # 4. Final fallback: Fear & Greed Index (market-wide sentiment)
        return self.get_fear_greed_score()

    def _get_twitter_sentiment(self, symbol: str) -> float:
        """
        Get sentiment from Twitter scraping results.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')

        Returns:
            float or None: Sentiment score if available and recent, None otherwise
        """
        import json

        # Check cache (15 min)
        if self._twitter_cache and self._twitter_cache_timestamp:
            age = datetime.now() - self._twitter_cache_timestamp
            if age.total_seconds() < 900 and symbol in self._twitter_cache:
                score = self._twitter_cache[symbol]
                cprint(f"[Sentiment] Twitter ({symbol}): {score:.2f} (cached)", "cyan")
                return score

        # Find most recent Twitter sentiment file
        if not self.twitter_dir.exists():
            return None

        try:
            sentiment_files = sorted(self.twitter_dir.glob("sentiment_*.json"), reverse=True)
            if not sentiment_files:
                return None

            # Read most recent file
            with open(sentiment_files[0], 'r') as f:
                data = json.load(f)

            # Check if data is recent (< 2 hours)
            if symbol in data:
                entry = data[symbol]
                if 'timestamp' in entry and 'sentiment_score' in entry:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    age = datetime.now() - timestamp
                    if age.total_seconds() < 7200:  # 2 hours
                        score = entry['sentiment_score']

                        # Update cache
                        self._twitter_cache = {k: v.get('sentiment_score', 0) for k, v in data.items() if isinstance(v, dict)}
                        self._twitter_cache_timestamp = datetime.now()

                        cprint(f"[Sentiment] Twitter ({symbol}): {score:.2f} ({entry.get('tweets_found', 0)} tweets)", "cyan")
                        return score

        except Exception as e:
            cprint(f"[Sentiment] Twitter read error: {e}", "yellow")

        return None

    def get_sentiment_trend(self, symbol: str, hours: int = 24) -> float:
        """
        Calculate the sentiment trend over the specified time period.

        Args:
            symbol: Crypto symbol
            hours: Number of hours to look back

        Returns:
            float: Trend value (-1 to +1), positive = improving, negative = declining
        """
        if not self.history_path.exists():
            return 0.0

        try:
            df = pd.read_csv(self.history_path)
            if df.empty or 'timestamp' not in df.columns:
                return 0.0

            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df['timestamp'] > cutoff]

            if len(df) < 2:
                return 0.0

            # Calculate trend as difference between recent and older sentiment
            first_half = df.head(len(df) // 2)['sentiment_score'].mean()
            second_half = df.tail(len(df) // 2)['sentiment_score'].mean()

            trend = second_half - first_half
            # Clamp to -1 to +1
            return max(-1.0, min(1.0, trend))

        except Exception:
            return 0.0

    def get_sentiment_volatility(self, symbol: str, hours: int = 24) -> float:
        """
        Calculate sentiment volatility (standard deviation) over time.

        Args:
            symbol: Crypto symbol
            hours: Number of hours to look back

        Returns:
            float: Volatility score (0 to 1)
        """
        if not self.history_path.exists():
            return 0.0

        try:
            df = pd.read_csv(self.history_path)
            if df.empty or 'sentiment_score' not in df.columns:
                return 0.0

            # Calculate standard deviation
            std = df['sentiment_score'].std()
            # Normalize to 0-1 range (assuming max std of 1)
            return min(1.0, std)

        except Exception:
            return 0.0

    def has_recent_data(self, max_age_hours: float = 2.0) -> bool:
        """
        Check if sentiment data is recent enough to be useful.

        Args:
            max_age_hours: Maximum age of data in hours

        Returns:
            bool: True if data is recent, False otherwise
        """
        # Fear & Greed is always available as fallback
        if self.get_fear_greed_score() != 0.0:
            return True

        if not self.history_path.exists():
            return False

        try:
            df = pd.read_csv(self.history_path)
            if df.empty or 'timestamp' not in df.columns:
                return False

            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            age = datetime.now() - last_timestamp

            return age.total_seconds() < (max_age_hours * 3600)

        except Exception:
            return False

    def get_fear_greed_score(self) -> float:
        """
        Get the Fear & Greed Index from Alternative.me API.
        Cached for 30 minutes to avoid rate limits.

        Returns:
            float: Score from -1 (extreme fear) to +1 (extreme greed)
        """
        # Check cache (30 min)
        if self._fear_greed_cache is not None and self._fear_greed_timestamp:
            age = datetime.now() - self._fear_greed_timestamp
            if age.total_seconds() < 1800:  # 30 minutes
                return self._fear_greed_cache

        try:
            response = requests.get(
                "https://api.alternative.me/fng/",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    # Fear & Greed is 0-100, convert to -1 to +1
                    # 0 = Extreme Fear (-1), 50 = Neutral (0), 100 = Extreme Greed (+1)
                    value = int(data['data'][0]['value'])
                    score = (value - 50) / 50  # Convert 0-100 to -1 to +1

                    # Cache the result
                    self._fear_greed_cache = score
                    self._fear_greed_timestamp = datetime.now()

                    classification = data['data'][0].get('value_classification', 'Unknown')
                    cprint(f"[Sentiment] Fear & Greed: {value} ({classification}) -> Score: {score:.2f}", "cyan")

                    return score

        except Exception as e:
            cprint(f"[Sentiment] Fear & Greed API error: {e}", "yellow")

        return 0.0
