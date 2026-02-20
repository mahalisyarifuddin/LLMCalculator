
from playwright.sync_api import sync_playwright
import os

def run():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    html_path = os.path.join(project_root, "LLMCalculator.html")
    screenshots_dir = os.path.join(project_root, "screenshots")

    # Ensure screenshots directory exists
    os.makedirs(screenshots_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # Desktop
        print(f"Taking desktop screenshot of {html_path}...")
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        page.goto(f'file://{html_path}')
        desktop_path = os.path.join(screenshots_dir, 'desktop.png')
        page.screenshot(path=desktop_path, full_page=True)
        print(f"Saved to {desktop_path}")

        # Mobile
        print(f"Taking mobile screenshot...")
        page_mobile = browser.new_page(viewport={'width': 375, 'height': 667})
        page_mobile.goto(f'file://{html_path}')
        mobile_path = os.path.join(screenshots_dir, 'mobile.png')
        page_mobile.screenshot(path=mobile_path, full_page=True)
        print(f"Saved to {mobile_path}")

        browser.close()

if __name__ == "__main__":
    run()
