"""
Generate PNG screenshots from HTML diagrams using Playwright.
"""
from playwright.sync_api import sync_playwright
import os

base = r"c:\Users\mntalhakilic\OneDrive - Microsoft\Desktop\Research\substack\01-attention-is-all-you-need"
img_dir = os.path.join(base, "images")
out_dir = os.path.join(base, "images", "png")
os.makedirs(out_dir, exist_ok=True)

files = [
    ("01-architecture.html", "01-architecture.png", 1100, 1400),
    ("02-attention-mechanism.html", "02-attention-mechanism.png", 1100, 1600),
    ("03-positional-encoding.html", "03-positional-encoding.png", 900, 800),
    ("04-why-parallel.html", "04-why-parallel.png", 1000, 1400),
    ("05-tensor-shapes.html", "05-tensor-shapes.png", 1000, 2200),
    ("06-training-results.html", "06-training-results.png", 900, 1600),
    ("07-concrete-example.html", "07-concrete-example.png", 1100, 5600),
    ("08-layer-flow.html", "08-layer-flow.png", 1100, 2000),
    ("09-llama-architecture.html", "09-llama-architecture.png", 1100, 2400),
    ("10-interactive-deep-dive.html", "10-interactive-deep-dive.png", 900, 3200),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    
    for html_file, png_file, w, h in files:
        html_path = os.path.join(img_dir, html_file)
        out_path = os.path.join(out_dir, png_file)
        file_url = "file:///" + html_path.replace("\\", "/")
        
        print(f"Rendering {html_file} ({w}x{h})...")
        page = browser.new_page(viewport={"width": w, "height": h})
        page.goto(file_url, wait_until="networkidle")
        page.screenshot(path=out_path, full_page=True)
        page.close()
        
        if os.path.exists(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  OK  {png_file} ({size_kb:.0f} KB)")
        else:
            print(f"  FAIL")
    
    browser.close()

print(f"\nAll images saved to: {out_dir}")
