from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1000})
    page.goto('http://127.0.0.1:8501')
    page.get_by_text('Skenario demo', exact=True).wait_for(timeout=60000)
    for name, status in [('Kondisi stabil','NORMAL'), ('pH rendah','WASPADA'), ('Beberapa parameter bermasalah','KRITIS'), ('Kondisi stabil','NORMAL')]:
        if name != 'Kondisi stabil' or 'KRITIS' in page.locator('.headbar').inner_text():
            page.get_by_role('combobox').first.click()
            page.get_by_role('option',name=name,exact=True).click()
        page.wait_for_function('(s)=>document.querySelector(".headbar")?.innerText.includes(s)',arg=status)
        print(name, page.locator('.headbar').inner_text(), page.locator('.headbar').evaluate('(e)=>getComputedStyle(e).backgroundColor'))
    browser.close()
