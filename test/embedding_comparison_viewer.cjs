const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/coll/.npm/_npx/420ff84f11983ee5/node_modules/playwright');
const assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 const out=path.resolve(__dirname,'../output/embedding_comparison');
 await page.goto('file://'+path.join(out,'demo.html'));
 assert.equal(await page.locator('#cards .card').count(),4);
 assert.equal(await page.locator('canvas').count(),2);
 assert.equal(await page.locator('#metrics tr').count(),4);
 for(const cohort of ['natural','topics']){
  await page.selectOption('#cohort',cohort);
  for(const method of ['kmeans','tree']){
   await page.selectOption('#method',method);
   await page.selectOption('#color','group');
   for(const model of ['current','arctic']){
    await page.locator('#panel-'+model+' .group').selectOption({index:1});
    assert.equal(await page.locator('#panel-'+model+' .examples .post').count(),5);
   }
   await page.selectOption('#query','1');
   assert.equal(await page.locator('#neighbors .post').count(),10);
  }
 }
 await page.selectOption('#color','topic');
 await page.locator('#panel-current .group').selectOption('all');
 // Click a real projected point rather than an arbitrary blank canvas pixel.
 const target=await page.evaluate(()=>{const c=cohort(),r=c.models.current,canvas=$('panel-current').querySelector('canvas'),p=transform(r.projection.points,canvas)(r.projection.points[0]),b=canvas.getBoundingClientRect();return {x:b.x+p[0]*b.width/canvas.width,y:b.y+p[1]*b.height/canvas.height}});
 await page.mouse.click(target.x,target.y);
 assert.equal(await page.locator('#panel-current .selected .post').count(),1);
 await page.locator('#panel-current .group').selectOption('all');
 await page.locator('#panel-arctic .group').selectOption('all');
 await page.screenshot({path:path.join(out,'preview.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 assert.deepEqual(errors,[]);
 // Post content never becomes active markup.
 assert.equal(await page.evaluate(()=>{const p=post({author:'x',topic:'',text:'<img src=x onerror=alert(1)>',input_text:'<img src=x onerror=alert(1)>'});return p.querySelectorAll('img').length}),0);
 fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({cohorts:2,methods:2,point_inspection:true,mobile:true,errors},null,2)+'\n');
 await browser.close();console.log('Embedding comparison viewer: cohort/method controls, neighbors, point inspection, escaping and mobile passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
