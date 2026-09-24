const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/coll/.npm/_npx/420ff84f11983ee5/node_modules/playwright');
const assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1450,height:1080}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));const out=path.resolve(__dirname,'../output/density_ablation');
 await page.goto('file://'+path.join(out,'demo.html'));
 assert.equal(await page.locator('#outcomes tr').count(),6);
 for(const dataset of ['Bluesky 100,000','Axis-aligned 2D topics','Correlated 2D topics','Correlated 32D topics']){
  await page.selectOption('#case',dataset);
  for(const world of ['Fermat: density + proposals','Fermat: proposals only','Euclidean: proposals only']){
   await page.selectOption('#world',world);
   assert.equal(await page.locator('#outcomes tr').count(),world.includes('density +')?6:4);
   assert.equal(await page.locator('#directPaired p').count(),world.includes('density +')?2:0);
   for(const radius of ['0.05','0.1'])await page.selectOption('#radius',radius);
  }
 }
 await page.selectOption('#world','Fermat: density + proposals');
 await page.selectOption('#method','Actual posts + PPCA density');
 await page.locator('#step').fill('40');assert.equal(await page.locator('#stepValue').textContent(),'40');
 await page.locator('#reveal').uncheck();await page.locator('#reveal').check();
 await page.click('#play');await page.waitForTimeout(400);await page.click('#play');
 assert.equal(await page.locator('#play').textContent(),'Play');
 await page.selectOption('#case','Bluesky 100,000');await page.selectOption('#radius','0.05');
 await page.screenshot({path:path.join(out,'preview.png'),fullPage:true});
 await page.setViewportSize({width:480,height:900});await page.waitForTimeout(150);
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({desktop:true,mobile:true,controls:true,methods:6,cases:4,errors},null,2)+'\n');
 await browser.close();console.log('Density ablation viewer: desktop, mobile, playback and selectors passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
