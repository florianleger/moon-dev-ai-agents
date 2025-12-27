"""
Twitter Login with Persistent Browser Profile
Run this script ONCE to login manually, then cookies are saved for reuse.
Uses a persistent browser context that saves state between sessions.
"""
import asyncio
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import os
import json
from termcolor import cprint
from pathlib import Path

load_dotenv()

# Persistent profile directory
PROFILE_DIR = Path(__file__).parent.parent.parent / "browser_data"

async def login_twitter_persistent():
    async with async_playwright() as p:
        cprint("\n=== Twitter Login with Persistent Profile ===\n", "cyan")
        cprint("This browser profile will be SAVED between sessions.", "green")
        cprint("You only need to login ONCE manually.\n", "green")

        # Create profile directory
        PROFILE_DIR.mkdir(exist_ok=True)

        # Launch with persistent context (saves cookies, localStorage, etc.)
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR / "chromium_profile"),
            headless=False,
            slow_mo=100,
            viewport={'width': 1280, 'height': 800},
            locale='en-US',
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-first-run',
                '--no-default-browser-check'
            ]
        )

        page = await context.new_page()

        try:
            # Check if already logged in
            cprint("Checking if already logged in...", "yellow")
            await page.goto('https://x.com/home', timeout=60000, wait_until='networkidle')
            await asyncio.sleep(3)

            current_url = page.url
            cprint(f"Current URL: {current_url}", "white")

            if 'home' in current_url and 'login' not in current_url:
                cprint("\nALREADY LOGGED IN!", "green")
                cprint("Your browser profile has saved cookies.", "green")
            else:
                cprint("\nNot logged in yet. Please login MANUALLY in the browser.", "yellow")
                cprint("The browser will stay open. Take your time.", "yellow")
                cprint("\nWaiting for you to complete login...", "cyan")

                # Navigate to login page
                await page.goto('https://x.com/login', timeout=60000, wait_until='networkidle')

                # Wait for user to manually login (up to 5 minutes)
                for i in range(60):
                    await asyncio.sleep(5)
                    current_url = page.url
                    if 'home' in current_url and 'login' not in current_url:
                        cprint("\nLOGIN DETECTED! Saving session...", "green")
                        break
                    if i % 12 == 0 and i > 0:  # Every minute
                        cprint(f"Still waiting... ({i//12} min)", "white")
                else:
                    cprint("\nTimeout waiting for login. Try again.", "red")

            # Save cookies to cookies.json for other scripts
            cookies = await context.cookies()
            cookies_path = Path(__file__).parent.parent.parent / "cookies.json"
            with open(cookies_path, 'w') as f:
                json.dump(cookies, f, indent=2)

            auth_cookies = [c for c in cookies if 'auth' in c.get('name', '').lower() or c.get('name') == 'ct0']

            if auth_cookies:
                cprint(f"\nSaved {len(cookies)} cookies ({len(auth_cookies)} auth cookies)", "green")
                cprint(f"Cookies saved to: {cookies_path}", "green")
                cprint("\nYou can now close the browser.", "cyan")
            else:
                cprint(f"\nSaved {len(cookies)} cookies (no auth cookies found)", "yellow")
                cprint("Login may not have completed successfully.", "yellow")

            # Keep browser open for user to verify
            cprint("\nPress Enter in the terminal to close the browser...", "cyan")
            await asyncio.sleep(2)

        except Exception as e:
            cprint(f"\nError: {e}", "red")

        finally:
            # Don't close context - let user close it manually or press Enter
            input("\nPress Enter to close browser...")
            await context.close()
            cprint("Browser closed. Profile saved.", "white")

if __name__ == "__main__":
    cprint("\n" + "="*50, "cyan")
    cprint("TWITTER PERSISTENT LOGIN", "cyan")
    cprint("="*50 + "\n", "cyan")
    cprint("This script uses a SAVED browser profile.", "white")
    cprint("Login once manually, and it will remember you.\n", "white")
    asyncio.run(login_twitter_persistent())
