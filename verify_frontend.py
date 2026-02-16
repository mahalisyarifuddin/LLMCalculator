from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        file_path = os.path.abspath("LLMCalculator.html")
        page.goto(f"file://{file_path}")

        # Wait for calculation
        page.wait_for_selector("#resultValue")

        # 1. Verify Default State (24GB VRAM, 10% Overhead)
        vram_val = page.locator("#vramValue").inner_text()
        overhead_val = page.locator("#overheadValue").inner_text()
        print(f"Initial VRAM: {vram_val}, Overhead: {overhead_val}")

        result_val = page.locator("#resultValue").inner_text()
        result_cat = page.locator("#resultCategory").inner_text()
        print(f"Initial Result: {result_val}, {result_cat}")

        # Verify "Estimated Architecture" text is present
        assert "Estimated Architecture" in result_cat

        # 2. Change VRAM to 80GB
        page.locator("#vramSlider").fill("80")
        page.locator("#vramSlider").dispatch_event("input")
        page.wait_for_timeout(500) # Wait for update

        vram_val_80 = page.locator("#vramValue").inner_text()
        result_val_80 = page.locator("#resultValue").inner_text()
        print(f"80GB VRAM Result: {result_val_80}")

        # 3. Change Overhead to 50%
        page.locator("#overheadSlider").fill("50")
        page.locator("#overheadSlider").dispatch_event("input")
        page.wait_for_timeout(500)

        overhead_val_50 = page.locator("#overheadValue").inner_text()
        result_val_50 = page.locator("#resultValue").inner_text()
        print(f"50% Overhead Result: {result_val_50}")

        # Verify result decreased
        # Extract number from "XXB"
        def parse_b(s):
            return float(s.replace("B", ""))

        if parse_b(result_val_50) < parse_b(result_val_80):
            print("Verified: Overhead reduction works.")
        else:
            print("Error: Overhead reduction did not reduce params.")

        # 4. Take Screenshot
        page.screenshot(path="/home/jules/verification/verification.png")
        print("Screenshot saved.")

        browser.close()

if __name__ == "__main__":
    run()
