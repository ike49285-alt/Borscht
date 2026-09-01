// Open the single-file build, touch nothing, and record what a viewer sees.
//
// The browser check proves the bundle works. This answers a different question:
// whether it is worth watching. A page that runs correctly but takes ten
// minutes to leave founding scatter has failed at the only thing it is for.
//
// Run it after changing the opening seed, scale or speed. Under software
// rendering in this container the world should reach its radiation -- a few
// thousand animals across dozens of species -- inside a minute; on real
// hardware, sooner. Note that the frame rate here is a floor, not a forecast:
// SwiftShader rasterises in software and dominates the per-step cost.
//
// Expects out/web/bundle/index.html, which `check-web.mjs --bundle` writes.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
const dir = 'out/web/bundle';
const s = spawn('python3',['-m','http.server','-d',dir,'8124','--bind','127.0.0.1'],{stdio:'ignore'});
process.on('exit',()=>s.kill());
for(let i=0;i<50;i++){try{if((await fetch('http://127.0.0.1:8124/index.html')).ok)break;}catch{await new Promise(r=>setTimeout(r,100));}}
const b = await chromium.launch({executablePath:'/opt/pw-browsers/chromium',args:['--use-gl=swiftshader','--enable-unsafe-swiftshader']});
const p = await b.newPage({viewport:{width:1400,height:900}});
await p.goto('http://127.0.0.1:8124/index.html',{waitUntil:'load'});
const t0 = Date.now();
for (const at of [5,15,30,60,120]) {
  while (Date.now()-t0 < at*1000) await p.waitForTimeout(500);
  const r = {};
  for (const k of ['tick','plants','animals','species','carn','ms','fps'])
    r[k]=(await p.textContent('#h-'+k)).trim();
  console.log(`${String(at).padStart(3)}s  tick ${r.tick.padStart(6)}  plants ${r.plants.padStart(6)}  animals ${r.animals.padStart(6)}  spp ${r.species.padStart(9)}  carn ${r.carn.padStart(5)}  ${r.ms}ms  ${r.fps}fps`);
}
await p.screenshot({path:'out/web/live.png'});
await b.close(); s.kill();
