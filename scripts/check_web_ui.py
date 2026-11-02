from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8000')
    page.locator('#status strong').wait_for()
    assert 'NORMAL' in page.locator('#status').inner_text()
    page.select_option('#scenario', 'ph')
    page.wait_for_function("document.querySelector('#status').textContent.includes('WASPADA')")
    page.select_option('#scenario', 'critical')
    page.wait_for_function("document.querySelector('#status').textContent.includes('KRITIS')")
    page.select_option('#scenario', 'stable')
    page.wait_for_function("document.querySelector('#status').textContent.includes('NORMAL')")
    page.screenshot(path=str(ROOT / 'web-ui-check.png'), full_page=True)
    page.click('[data-page=lab]')
    page.click('#train')
    page.locator('#lab-result:not([hidden])').wait_for(timeout=60000)
    page.select_option('#lab-model', 'Random Forest')
    with page.expect_download() as download:
        page.click('#download-results')
    assert 'SIMULASI' in download.value.suggested_filename
    page.click('[data-page=upload]')
    page.set_input_files('#file', str(ROOT / 'data/biogas_data.csv'))
    page.check('#confirmed')
    page.click('button[type=submit]')
    page.wait_for_function("document.querySelector('#source-note').textContent.includes('DATA UNGGAHAN')")
    assert 'Belum tersedia' in page.locator('#cards').inner_text()
    page.set_viewport_size({'width':390, 'height':844})
    page.wait_for_timeout(500)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
    page.screenshot(path=str(ROOT / 'web-mobile-ui-check.png'), full_page=True)
    assert not errors, errors
    print('PASS: scenarios, training, model selection, CSV download/upload, mobile width; no browser errors')
    browser.close()
