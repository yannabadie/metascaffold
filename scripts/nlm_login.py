"""Custom NotebookLM login script with extended timeout.

Workaround for corporate networks where the default 30s timeout is too short.
Usage: uv run python scripts/nlm_login.py
"""

from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    storage_path = Path.home() / ".notebooklm" / "storage_state.json"
    browser_profile = Path.home() / ".notebooklm" / "browser_profile"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    browser_profile.mkdir(parents=True, exist_ok=True)

    print("Opening browser for Google login (timeout: 120s)...")
    print(f"Using profile: {browser_profile}")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(browser_profile),
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--password-store=basic",
            ],
            ignore_default_args=["--enable-automation"],
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://notebooklm.google.com/", timeout=120_000)

        print()
        print("Instructions:")
        print("  1. Complete the Google login in the browser window")
        print("  2. Wait until you see the NotebookLM homepage")
        print("  3. Press ENTER here to save and close")
        print()

        input("[Press ENTER when logged in] ")

        context.storage_state(path=str(storage_path))
        context.close()

    print(f"\nAuthentication saved to: {storage_path}")


if __name__ == "__main__":
    main()
