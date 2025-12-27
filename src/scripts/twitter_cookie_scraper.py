"""
Twitter Scraper using Cookie Authentication
No login required - uses pre-authenticated cookies from .env
"""
import asyncio
import os
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from termcolor import cprint

load_dotenv()

# Use Patchright for stealth
from patchright.async_api import async_playwright

# Get cookies from .env
TWITTER_AUTH_TOKEN = os.getenv('TWITTER_AUTH_TOKEN', '')
TWITTER_CT0 = os.getenv('TWITTER_CT0', '')

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
DATA_DIR = Path(__file__).parent.parent / "data" / "twitter_sentiment"


def get_twitter_cookies():
    """Build Twitter cookies list for Playwright"""
    return [
        {
            "name": "auth_token",
            "value": TWITTER_AUTH_TOKEN,
            "domain": ".x.com",
            "path": "/",
            "secure": True,
            "httpOnly": True,
            "sameSite": "None"
        },
        {
            "name": "ct0",
            "value": TWITTER_CT0,
            "domain": ".x.com",
            "path": "/",
            "secure": True,
            "httpOnly": False,
            "sameSite": "Lax"
        }
    ]


async def test_twitter_access():
    """Test if cookies work for Twitter access"""

    if not TWITTER_AUTH_TOKEN or not TWITTER_CT0:
        cprint("ERROR: Twitter cookies not found in .env", "red")
        cprint("Add TWITTER_AUTH_TOKEN and TWITTER_CT0 to your .env file", "yellow")
        return False

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        cprint("\n=== Testing Twitter Cookie Access ===\n", "cyan")

        browser = await p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled']
        )

        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        # Add cookies BEFORE navigating
        cprint("Adding authentication cookies...", "yellow")
        await context.add_cookies(get_twitter_cookies())

        page = await context.new_page()

        try:
            cprint("Navigating to Twitter home...", "yellow")
            await page.goto('https://x.com/home', timeout=60000, wait_until='domcontentloaded')
            await asyncio.sleep(3)

            current_url = page.url
            await page.screenshot(path=str(SCREENSHOTS_DIR / "cookie_test.png"))

            if 'home' in current_url and 'login' not in current_url:
                cprint("\nSUCCESS! Cookies are valid - logged in!", "green")
                cprint(f"Screenshot saved: {SCREENSHOTS_DIR / 'cookie_test.png'}", "white")

                # Keep browser open for 5 seconds to verify
                await asyncio.sleep(5)
                await browser.close()
                return True
            else:
                cprint(f"\nCookies may be invalid. URL: {current_url}", "red")
                cprint(f"Screenshot saved: {SCREENSHOTS_DIR / 'cookie_test.png'}", "white")
                await asyncio.sleep(5)
                await browser.close()
                return False

        except Exception as e:
            cprint(f"Error: {e}", "red")
            await page.screenshot(path=str(SCREENSHOTS_DIR / "cookie_error.png"))
            await browser.close()
            return False


async def scrape_crypto_sentiment(symbols=['BTC', 'ETH', 'SOL']):
    """Scrape Twitter sentiment for crypto symbols"""

    if not TWITTER_AUTH_TOKEN or not TWITTER_CT0:
        cprint("ERROR: Twitter cookies not found", "red")
        return None

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    async with async_playwright() as p:
        cprint("\n=== Scraping Twitter Crypto Sentiment ===\n", "cyan")

        browser = await p.chromium.launch(
            headless=True,  # Run headless for scraping
            args=['--disable-blink-features=AutomationControlled']
        )

        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )

        await context.add_cookies(get_twitter_cookies())
        page = await context.new_page()

        for symbol in symbols:
            cprint(f"Searching for ${symbol}...", "yellow")

            try:
                # Search for crypto symbol
                search_url = f'https://x.com/search?q=%24{symbol}&src=typed_query&f=live'
                await page.goto(search_url, timeout=60000, wait_until='domcontentloaded')
                await asyncio.sleep(3)

                # Extract tweets
                tweets = await page.query_selector_all('[data-testid="tweet"]')

                tweet_texts = []
                for tweet in tweets[:20]:  # Get first 20 tweets
                    try:
                        text_el = await tweet.query_selector('[data-testid="tweetText"]')
                        if text_el:
                            text = await text_el.inner_text()
                            tweet_texts.append(text)
                    except:
                        pass

                # Simple sentiment analysis
                bullish_words = ['bull', 'moon', 'pump', 'buy', 'long', 'breakout', 'bullish', 'ath', 'green']
                bearish_words = ['bear', 'dump', 'sell', 'short', 'crash', 'bearish', 'red', 'rekt']

                bullish_count = 0
                bearish_count = 0

                for text in tweet_texts:
                    text_lower = text.lower()
                    bullish_count += sum(1 for word in bullish_words if word in text_lower)
                    bearish_count += sum(1 for word in bearish_words if word in text_lower)

                total = bullish_count + bearish_count
                if total > 0:
                    sentiment_score = (bullish_count - bearish_count) / total
                else:
                    sentiment_score = 0

                results[symbol] = {
                    'tweets_found': len(tweet_texts),
                    'bullish_signals': bullish_count,
                    'bearish_signals': bearish_count,
                    'sentiment_score': sentiment_score,
                    'timestamp': datetime.now().isoformat()
                }

                cprint(f"  {symbol}: {len(tweet_texts)} tweets, sentiment: {sentiment_score:.2f}",
                       "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "white")

            except Exception as e:
                cprint(f"  Error scraping {symbol}: {e}", "red")
                results[symbol] = {'error': str(e)}

        await browser.close()

    # Save results
    output_file = DATA_DIR / f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    cprint(f"\nResults saved to: {output_file}", "green")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--scrape":
        # Scrape sentiment
        symbols = sys.argv[2:] if len(sys.argv) > 2 else ['BTC', 'ETH', 'SOL']
        asyncio.run(scrape_crypto_sentiment(symbols))
    else:
        # Test cookie access
        asyncio.run(test_twitter_access())
