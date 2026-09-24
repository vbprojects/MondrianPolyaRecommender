const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/home/coll/.npm/_npx/420ff84f11983ee5/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless:true, args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1450,height:1050}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const out=path.resolve(__dirname,'../output/streaming_codecs');
  await page.goto('file://'+path.join(out,'demo.html'));
  assert.equal(await page.locator('#rows tr').count(),12);
  assert.equal(await page.locator('#case option').count(),6);
  for (const data of ['Bluesky 100,000','Stable rank-four manifold','Rotating, long retention']) {
    await page.selectOption('#case',data);
    for (const metric of ['recall10','local_proposal_recall10','nrmse','membership_error','ig_regret_bits']) {
      await page.selectOption('#metric',metric);
      assert.equal(await page.locator('#rows tr').count(),12);
    }
  }
  await page.selectOption('#phase','merged');
  await page.selectOption('#frameMethod','Inherited basis + innovations');
  await page.locator('#day').fill('3');
  assert.equal(await page.locator('#dayValue').textContent(),'3');
  await page.locator('#legend input').first().uncheck();
  await page.selectOption('#case','Bluesky 100,000');
  await page.selectOption('#metric','local_proposal_recall10');
  await page.selectOption('#phase','grown');
  await page.locator('#legend input').first().check();
  await page.screenshot({path:path.join(out,'preview.png'),fullPage:true});
  await page.setViewportSize({width:480,height:900});
  await page.waitForTimeout(150);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({rows:12,cases:6,desktop:true,mobile:true,errors},null,2)+'\n');
  await browser.close();console.log('Streaming codec viewer: desktop, mobile and controls passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
