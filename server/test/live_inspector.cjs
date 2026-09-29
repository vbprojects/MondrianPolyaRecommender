// Run against an inspector: INSPECTOR_URL=http://127.0.0.1:8091 node server/test/live_inspector.cjs
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(process.env.INSPECTOR_URL || 'http://127.0.0.1:8091');
    await page.waitForSelector('#treemap g');
    await page.locator('#pause').click();
    await page.locator('#treemap g').first().click();
    await page.waitForSelector('.post');
    assert.equal(await page.locator('.post').count(),6);
    assert.ok(await page.locator('.keyword').count()>0);
    await page.screenshot({path:'output/live_inspector/desktop.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:'output/live_inspector/mobile.png',fullPage:true});
    // Imported text must remain inert, including malicious permalink schemes.
    await page.route('**/api/leaf?*',async route=>{
      const response=await route.fetch(),d=await response.json();
      if (!response.ok()) return route.fulfill({response});
      for (const p of d.posts) {
        p.text='<img src=x onerror="window.injected=true">';
        p.permalink='javascript:window.injected=true';
      }
      await route.fulfill({response,json:d});
    });
    await page.locator('#refresh').click();
    await page.waitForFunction(()=>document.querySelector('.post p')?.textContent.startsWith('<img'));
    assert.equal(await page.locator('.post img').count(),0);
    assert.equal(await page.locator('.post a').count(),0);
    assert.equal(await page.evaluate(()=>window.injected),undefined);
    await page.locator('#tree > details > summary button').click();
    assert.match(await page.locator('#detail-body').textContent(),/Dimension/);
    await page.locator('#root').click();
    assert.match(await page.locator('#breadcrumb').textContent(),/100.00%/);
    assert.deepEqual(errors,[]);
    console.log('PASS: live leaf selection, keywords, examples, mobile layout, refresh, hierarchy, inert text/links');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
