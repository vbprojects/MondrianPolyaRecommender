const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/coll/.npm/_npx/420ff84f11983ee5/node_modules/playwright');
const assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1040}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  const out=path.resolve(__dirname,'../output/disk_ablation');
  await page.goto('file://'+path.join(out,'demo.html'));
  assert.equal(await page.locator('#checks').textContent(),'36');
  for(const layout of ['append','leaf']){
    await page.selectOption('#layout',layout);
    for(const budget of ['32','256','1024','4096']){
      await page.selectOption('#budget',budget);
      for(const leaves of ['1','4','16']){
        await page.selectOption('#leaves',leaves);
        for(const condition of ['application_cold','immediate_repeat','mixed','file_cache_eviction_hint','continuous_uniform','continuous_topic_burst','continuous_cold_revisit']){
          await page.selectOption('#condition',condition);
          assert.equal(await page.locator('#latency tbody tr').count(),6);
          assert(!/NaN|undefined/.test(await page.locator('main').innerText()));
        }
      }
    }
  }
  await page.selectOption('#budget','256');await page.selectOption('#leaves','4');
  await page.selectOption('#condition','file_cache_eviction_hint');
  await page.screenshot({path:path.join(out,'preview.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({desktop:true,mobile:true,control_combinations:168,errors},null,2)+'\n');
  await browser.close();console.log('Disk ablation viewer: 168 combinations, desktop and mobile passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
