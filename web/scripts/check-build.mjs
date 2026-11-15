import {readFile,stat} from 'node:fs/promises';
for(const file of ['public/index.html','public/app.js','public/live.js','public/data.json','api/handler.js','server/core.mjs','server/http.mjs','server/usage.mjs'])await stat(file);
const index=await readFile('public/index.html','utf8');
if(!index.includes('id="live"'))throw new Error('Live workspace missing');
const data=JSON.parse(await readFile('public/data.json','utf8'));if(data.cases.length!==90)throw new Error('Historical evaluation is incomplete');
console.log('Vercel app ready: live routes, public assets, and all 90 research records.');
