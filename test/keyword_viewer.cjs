const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/coll/.npm/_npx/420ff84f11983ee5/node_modules/playwright');
const assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 const out=path.resolve(__dirname,'../output/keywords');await page.goto('file://'+path.join(out,'demo.html'));
 assert.equal(await page.locator('#comparison tr').count(),4);
 let controls=0;
 for(const method of ['yake','hashtags']){
  await page.selectOption('#method',method);
  for(const stage of ['0','1','2']){
   await page.selectOption('#stage',stage);
   const values=await page.locator('#leaf option').evaluateAll(xs=>xs.map(x=>x.value));
   for(const leaf of [values[0],values[Math.floor(values.length/2)],values.at(-1)]){
    await page.selectOption('#leaf',leaf);
    for(const ranking of ['common','distinctive']){
     await page.selectOption('#ranking',ranking);assert(!/NaN|undefined/.test((await page.locator('#coverage').innerText())+(await page.locator('#comparison').innerText())));controls++;
    }
   }
  }
 }
 await page.selectOption('#method','yake');await page.selectOption('#stage','1');await page.screenshot({path:path.join(out,'preview.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({controls,desktop:true,mobile:true,errors},null,2)+'\n');
 await browser.close();console.log(`Keyword viewer: ${controls} controls passed`);
})().catch(e=>{console.error(e);process.exitCode=1});
