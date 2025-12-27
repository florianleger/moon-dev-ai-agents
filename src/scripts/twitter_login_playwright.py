"""
Twitter Login with Playwright - Interactive Version
Run this script to login to Twitter and generate cookies.json
Uses Firefox for better stealth
"""
import asyncio
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import os
import json
from termcolor import cprint
from pathlib import Path

load_dotenv()

USERNAME = os.getenv('TWITTER_USERNAME')
EMAIL = os.getenv('TWITTER_EMAIL')
PASSWORD = os.getenv('TWITTER_PASSWORD')

async def login_twitter():
    async with async_playwright() as p:
        cprint("\n=== Twitter Login with Playwright (Firefox) ===\n", "cyan")
        cprint(f"Username: {USERNAME}", "white")
        cprint(f"Email: {EMAIL}", "white")

        # Use Firefox - less likely to be detected
        browser = await p.firefox.launch(
            headless=False,
            slow_mo=500
        )

        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            locale='fr-FR'
        )

        page = await context.new_page()

        # Save screenshots to debug
        screenshot_dir = Path(__file__).parent / "screenshots"
        screenshot_dir.mkdir(exist_ok=True)

        try:
            cprint("Navigating to Twitter login...", "yellow")
            await page.goto('https://x.com/login', timeout=60000, wait_until='networkidle')
            await asyncio.sleep(3)
            await page.screenshot(path=str(screenshot_dir / "01_login_page.png"))

            # Wait for the login form to load
            cprint("Waiting for login form...", "yellow")
            await page.wait_for_selector('input[autocomplete="username"]', timeout=30000)

            # Enter username
            cprint(f"Entering username: {USERNAME}", "yellow")
            await page.fill('input[autocomplete="username"]', USERNAME)
            await asyncio.sleep(1)
            await page.screenshot(path=str(screenshot_dir / "02_username_entered.png"))

            # Click Next button
            cprint("Clicking Next...", "yellow")
            next_button = await page.query_selector('button:has-text("Suivant"), button:has-text("Next")')
            if next_button:
                await next_button.click()
            else:
                # Try pressing Enter
                await page.keyboard.press('Enter')
            await asyncio.sleep(3)
            await page.screenshot(path=str(screenshot_dir / "03_after_next.png"))

            # Check what page we're on
            current_html = await page.content()
            cprint(f"Current URL: {page.url}", "white")

            # Check for unusual activity / verification
            if "unusual" in current_html.lower() or "vérif" in current_html.lower():
                cprint("Twitter demande une vérification!", "red")
                cprint("Complétez manuellement dans le navigateur...", "yellow")
                await asyncio.sleep(60)
                await page.screenshot(path=str(screenshot_dir / "04_verification.png"))

            # Check for email verification
            email_input = await page.query_selector('input[data-testid="ocfEnterTextTextInput"]')
            if email_input:
                cprint(f"Email verification required, entering: {EMAIL}", "yellow")
                await email_input.fill(EMAIL)
                await asyncio.sleep(1)
                next_btn = await page.query_selector('button:has-text("Suivant"), button:has-text("Next")')
                if next_btn:
                    await next_btn.click()
                else:
                    await page.keyboard.press('Enter')
                await asyncio.sleep(3)
                await page.screenshot(path=str(screenshot_dir / "05_after_email.png"))

            # Wait for password field with multiple attempts
            cprint("Looking for password field...", "yellow")
            password_input = None
            for attempt in range(3):
                password_input = await page.query_selector('input[name="password"], input[type="password"]')
                if password_input:
                    break
                cprint(f"  Attempt {attempt + 1}/3 - waiting...", "white")
                await asyncio.sleep(3)
                await page.screenshot(path=str(screenshot_dir / f"06_looking_password_{attempt}.png"))

            if not password_input:
                cprint("Password field not found! Check screenshots.", "red")
                cprint("Waiting 60 seconds for manual login...", "yellow")
                await asyncio.sleep(60)
            else:
                # Enter password
                cprint("Entering password...", "yellow")
                await password_input.fill(PASSWORD)
                await asyncio.sleep(1)
                await page.screenshot(path=str(screenshot_dir / "07_password_entered.png"))

                # Click Login button
                cprint("Clicking Login...", "yellow")
                login_btn = await page.query_selector('button[data-testid="LoginForm_Login_Button"], button:has-text("Se connecter"), button:has-text("Log in")')
                if login_btn:
                    await login_btn.click()
                else:
                    await page.keyboard.press('Enter')

                cprint("Waiting for redirect...", "yellow")
                await asyncio.sleep(8)

            await page.screenshot(path=str(screenshot_dir / "08_final.png"))

            # Check current URL
            current_url = page.url
            cprint(f"Final URL: {current_url}", "white")

            # Save cookies
            cookies = await context.cookies()
            cookies_path = Path(__file__).parent.parent.parent / "cookies.json"
            with open(cookies_path, 'w') as f:
                json.dump(cookies, f, indent=2)

            auth_cookies = [c for c in cookies if 'auth' in c.get('name', '').lower() or c.get('name') == 'ct0']

            if auth_cookies or 'home' in current_url:
                cprint(f"\nLOGIN SUCCESSFUL!", "green")
                cprint(f"Saved {len(cookies)} cookies ({len(auth_cookies)} auth)", "green")
            else:
                cprint(f"\nLogin may have failed. Saved {len(cookies)} cookies.", "yellow")
                cprint(f"Check screenshots in: {screenshot_dir}", "yellow")

        except Exception as e:
            cprint(f"\nError during login: {e}", "red")
            await page.screenshot(path=str(screenshot_dir / "error.png"))
            cprint(f"Screenshot saved. Check: {screenshot_dir}", "yellow")
            cprint("\nWaiting 60 seconds for manual login...", "yellow")
            await asyncio.sleep(60)

            # Save cookies after wait
            cookies = await context.cookies()
            cookies_path = Path(__file__).parent.parent.parent / "cookies.json"
            with open(cookies_path, 'w') as f:
                json.dump(cookies, f, indent=2)
            cprint(f"Saved {len(cookies)} cookies to cookies.json", "green")

        finally:
            await asyncio.sleep(2)
            await browser.close()
            cprint("\nBrowser closed.", "white")

if __name__ == "__main__":
    asyncio.run(login_twitter())
