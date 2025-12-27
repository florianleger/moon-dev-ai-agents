"""
Twitter Stealth Login - Using Patchright (Undetected Playwright Fork)
Combines multiple anti-detection techniques:
1. Patchright - removes automation flags, avoids Runtime.enable detection
2. Device emulation - mimics real Desktop Chrome
3. Persistent context - saves cookies/state between sessions
4. playwright-stealth - patches navigator.webdriver and other fingerprints
5. Human-like behavior - random delays, realistic typing
"""
import asyncio
import random
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from termcolor import cprint

load_dotenv()

# Use Patchright instead of Playwright
from patchright.async_api import async_playwright

# playwright-stealth 2.0 uses Stealth class
try:
    from playwright_stealth import Stealth
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

# Configuration
PROFILE_DIR = Path(__file__).parent.parent.parent / "browser_data" / "stealth_profile"
COOKIES_PATH = Path(__file__).parent.parent.parent / "cookies.json"
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"

# Twitter credentials from .env
TWITTER_USERNAME = os.getenv('TWITTER_USERNAME', '')
TWITTER_PASSWORD = os.getenv('TWITTER_PASSWORD', '')


def human_delay(min_ms=500, max_ms=2000):
    """Random delay to mimic human behavior"""
    delay = random.randint(min_ms, max_ms) / 1000
    return delay


async def human_type(page, selector, text):
    """Type like a human with random delays between keystrokes"""
    element = await page.wait_for_selector(selector, timeout=30000)
    await element.click()
    await asyncio.sleep(human_delay(200, 500))

    for char in text:
        await page.keyboard.type(char)
        await asyncio.sleep(random.uniform(0.05, 0.15))  # 50-150ms between keys

    await asyncio.sleep(human_delay(300, 700))


async def login_twitter_stealth():
    """Login to Twitter using Patchright with stealth techniques"""

    # Create directories
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        cprint("\n" + "="*60, "cyan")
        cprint("TWITTER STEALTH LOGIN (Patchright + Anti-Detection)", "cyan")
        cprint("="*60 + "\n", "cyan")

        # Device emulation for Desktop Chrome
        device = p.devices.get("Desktop Chrome", {})

        cprint("Launching stealth browser with Patchright...", "yellow")

        # Launch Patchright browser with anti-detection flags
        browser = await p.chromium.launch(
            headless=False,  # MUST be headful for Twitter
            slow_mo=50,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--disable-extensions',
                '--disable-gpu',
                '--lang=en-US,en',
            ]
        )

        # Create context with device emulation and persistent storage
        context = await browser.new_context(
            storage_state=str(PROFILE_DIR / "state.json") if (PROFILE_DIR / "state.json").exists() else None,
            viewport={'width': 1280, 'height': 800},
            user_agent=device.get('user_agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            locale='en-US',
            timezone_id='America/New_York',
            geolocation={'latitude': 40.7128, 'longitude': -74.0060},
            permissions=['geolocation'],
            color_scheme='light',
            device_scale_factor=2,
        )

        page = await context.new_page()

        # Note: Patchright already applies most stealth patches automatically
        # navigator.webdriver is removed, automation flags are disabled
        cprint("Patchright stealth mode active.", "yellow")

        try:
            # Check if already logged in
            cprint("Checking login status...", "yellow")
            await asyncio.sleep(human_delay(1000, 2000))

            await page.goto('https://x.com/home', timeout=60000, wait_until='domcontentloaded')
            await asyncio.sleep(human_delay(2000, 4000))

            current_url = page.url
            cprint(f"Current URL: {current_url}", "white")

            # Take screenshot
            await page.screenshot(path=str(SCREENSHOTS_DIR / "01_initial.png"))

            if 'home' in current_url and 'login' not in current_url:
                cprint("\nALREADY LOGGED IN!", "green")
            else:
                cprint("\nNot logged in. Starting login process...", "yellow")

                # Navigate to login
                await page.goto('https://x.com/login', timeout=60000, wait_until='domcontentloaded')
                await asyncio.sleep(human_delay(2000, 4000))
                await page.screenshot(path=str(SCREENSHOTS_DIR / "02_login_page.png"))

                if TWITTER_USERNAME and TWITTER_PASSWORD:
                    cprint("Attempting automatic login with credentials...", "yellow")

                    # Enter username with human-like typing
                    cprint("Entering username...", "white")
                    await human_type(page, 'input[autocomplete="username"]', TWITTER_USERNAME)
                    await page.screenshot(path=str(SCREENSHOTS_DIR / "03_username_entered.png"))

                    # Click Next button
                    await asyncio.sleep(human_delay(500, 1000))
                    next_button = await page.wait_for_selector('text=Next', timeout=10000)
                    await next_button.click()
                    await asyncio.sleep(human_delay(2000, 3000))
                    await page.screenshot(path=str(SCREENSHOTS_DIR / "04_after_next.png"))

                    # Check for unusual activity prompt (phone/email verification)
                    unusual_activity = await page.query_selector('text=Enter your phone number or email address')
                    if unusual_activity:
                        cprint("\nTwitter requests additional verification (phone/email).", "yellow")
                        cprint("This is a security check, not bot detection.", "yellow")
                        cprint("Please complete verification manually in the browser.", "cyan")
                        await asyncio.sleep(60)  # Wait for manual intervention
                        await page.screenshot(path=str(SCREENSHOTS_DIR / "05_verification.png"))

                    # Enter password
                    try:
                        cprint("Entering password...", "white")
                        await human_type(page, 'input[name="password"]', TWITTER_PASSWORD)
                        await page.screenshot(path=str(SCREENSHOTS_DIR / "06_password_entered.png"))

                        # Click Log in button
                        await asyncio.sleep(human_delay(500, 1000))
                        login_button = await page.wait_for_selector('text=Log in', timeout=10000)
                        await login_button.click()
                        await asyncio.sleep(human_delay(3000, 5000))
                        await page.screenshot(path=str(SCREENSHOTS_DIR / "07_after_login.png"))

                    except Exception as e:
                        cprint(f"Could not find password field: {e}", "yellow")
                        cprint("Please complete login manually in the browser.", "cyan")
                        cprint("Waiting 2 minutes for manual login...", "cyan")
                        await asyncio.sleep(120)

                else:
                    cprint("\nNo credentials in .env file.", "yellow")
                    cprint("Please login MANUALLY in the browser.", "cyan")
                    cprint("The browser will stay open for 3 minutes.", "white")

                    # Wait for manual login
                    for i in range(36):  # 3 minutes
                        await asyncio.sleep(5)
                        current_url = page.url
                        if 'home' in current_url and 'login' not in current_url:
                            cprint("\nManual login detected!", "green")
                            break
                        if i % 12 == 0 and i > 0:
                            cprint(f"Still waiting... ({i//12} min)", "white")

            # Final check
            await asyncio.sleep(human_delay(2000, 3000))
            current_url = page.url
            await page.screenshot(path=str(SCREENSHOTS_DIR / "08_final.png"))

            if 'home' in current_url and 'login' not in current_url:
                cprint("\nLOGIN SUCCESSFUL!", "green")

                # Save storage state (cookies + localStorage)
                await context.storage_state(path=str(PROFILE_DIR / "state.json"))
                cprint(f"Session saved to: {PROFILE_DIR / 'state.json'}", "green")

                # Also export cookies for other scripts
                cookies = await context.cookies()
                with open(COOKIES_PATH, 'w') as f:
                    json.dump(cookies, f, indent=2)

                auth_cookies = [c for c in cookies if 'auth' in c.get('name', '').lower() or c.get('name') == 'ct0']
                cprint(f"Cookies saved to: {COOKIES_PATH} ({len(auth_cookies)} auth cookies)", "green")

            else:
                cprint(f"\nLogin may have failed. Current URL: {current_url}", "red")
                cprint("Check screenshots in src/scripts/screenshots/", "yellow")

            cprint("\nBrowser will close in 10 seconds...", "cyan")
            await asyncio.sleep(10)

        except Exception as e:
            cprint(f"\nError: {e}", "red")
            import traceback
            traceback.print_exc()
            await page.screenshot(path=str(SCREENSHOTS_DIR / "error.png"))

        finally:
            await context.close()
            await browser.close()
            cprint("Browser closed.", "white")


async def verify_stealth():
    """Verify stealth configuration by checking detection tests"""
    async with async_playwright() as p:
        cprint("\n=== Verifying Patchright Stealth Configuration ===\n", "cyan")

        browser = await p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await browser.new_context()
        page = await context.new_page()

        # Patchright already handles stealth - no need for extra patches

        # Check navigator.webdriver
        webdriver = await page.evaluate("navigator.webdriver")
        cprint(f"navigator.webdriver: {webdriver} (should be False or undefined)",
               "green" if not webdriver else "red")

        # Check Chrome detection
        chrome = await page.evaluate("window.chrome")
        cprint(f"window.chrome present: {chrome is not None}",
               "green" if chrome else "yellow")

        # Check for automation fingerprints
        cprint("\nLoading bot detection test page...", "yellow")
        await page.goto("https://bot.sannysoft.com/")
        await asyncio.sleep(5)

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(SCREENSHOTS_DIR / "stealth_test.png"))
        cprint(f"Stealth test screenshot saved to: {SCREENSHOTS_DIR / 'stealth_test.png'}", "green")

        await asyncio.sleep(5)
        await browser.close()


if __name__ == "__main__":
    import sys

    cprint("\n" + "="*60, "cyan")
    cprint("TWITTER STEALTH LOGIN", "cyan")
    cprint("Using Patchright (Undetected Playwright Fork)", "cyan")
    cprint("="*60 + "\n", "cyan")

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        cprint("Running stealth verification test...", "yellow")
        asyncio.run(verify_stealth())
    else:
        asyncio.run(login_twitter_stealth())
