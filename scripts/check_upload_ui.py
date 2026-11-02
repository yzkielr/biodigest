from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1000})
    page.goto('http://127.0.0.1:8501')
    page.get_by_text('Unggah CSV', exact=True).click(timeout=60000)
    page.get_by_test_id('stFileUploader').wait_for(timeout=60000)
    print('HERO:', page.locator('.upload-hero').count())
    print('TEXT:', page.locator('body').inner_text()[:3500])
    button = page.get_by_test_id('stFileUploaderDropzone').locator('button')
    print('BUTTON:', button.evaluate('(e)=>({background:getComputedStyle(e).backgroundColor,height:e.getBoundingClientRect().height})'))
    page.screenshot(path=str(Path(__file__).resolve().parents[1] / 'upload-ui-check.png'),full_page=True)
    browser.close()
