import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mutateUsage,reserveUsage} from '../server/usage.mjs';
import {handle} from '../server/http.mjs';
import {parseInput,retrieve,documents,validateOutput,riskSignals} from '../server/core.mjs';
const valid={intent:'playback_issue',confidence:.9,summary:'Audio pauses on Android.',answer:'Please confirm your device and app version.',cited_doc_ids:['KB-01'],needs_human:false,handoff_reason:'',clarification_question:''};
test('durable quota mutations enforce per-IP/global caps and duplicate protection',()=>{
 let state=null;const now=1794764000;
 for(let i=0;i<5;i++)state=mutateUsage(state,'client-a',`id-${i}`,now);
 assert.throws(()=>mutateUsage(state,'client-a','another',now),e=>e.code==='usage_limit');
 assert.throws(()=>mutateUsage(state,'client-a','id-0',now),e=>e.code==='duplicate_request');
 for(let i=0;i<10;i++)state=mutateUsage(state,`other-${i}`,`request-${i}`,now);
 assert.throws(()=>mutateUsage(state,'new-client','new-request',now),e=>e.code==='usage_limit');
 assert.doesNotThrow(()=>mutateUsage(state,'client-a','next-minute',now+61));
});
test('expired state is pruned and daily caps survive minute changes',()=>{
 let state={counters:{expired:{used:999,expires:1}},requests:{expired:1}};const now=1794764000;
 state=mutateUsage(state,'a','new',now);assert.equal(state.counters.expired,undefined);assert.equal(state.requests.expired,undefined);
 for(let i=1;i<30;i++)state=mutateUsage(state,'a',`new-${i}`,now+i*61);
 assert.throws(()=>mutateUsage(state,'a','over-day',now+1900),e=>e.code==='usage_limit');
});
test('conditional writes retry conflicts without losing concurrent increments',async()=>{
 let state=null,version=0,writes=0;
 const io={async read(){return {state:structuredClone(state),etag:version?String(version):null};},async write(path,body,options){writes++;await new Promise(r=>setTimeout(r,1));if((options.ifMatch||null)!==(version?String(version):null)){const error=new Error('conflict');error.name='BlobPreconditionFailedError';throw error;}state=JSON.parse(body);version++;}};
 await Promise.all(Array.from({length:4},(_,i)=>reserveUsage('same-ip',`request-${i}`,io,1794764000)));
 assert.equal(state.counters['global-minute:29912733'].used,4);assert.equal(Object.keys(state.requests).length,4);assert.ok(writes>=4);
});
test('storage outages fail closed instead of allowing unmetered inference',async()=>{
 await assert.rejects(reserveUsage('ip','id',{async read(){throw new Error('storage down');},async write(){throw new Error('must not write');}}));
});
test('Vercel handler returns live generated evidence and enforces security routing',async()=>{
 let called=0;const env={GEMINI_API_KEY:'unit-test',BLOB_READ_WRITE_TOKEN:'unit-test',RATE_LIMIT_SECRET:'unit-test'};
 const deps={async reserve(){called++;},async ready(){},async generate(){return {result:{...valid,intent:'account_security'},model:'test-model',usage:null};}};
 const request=new Request('https://hiver.test/api/support',{method:'POST',headers:{origin:'https://hiver.test','content-type':'application/json','x-vercel-forwarded-for':'192.0.2.1'},body:JSON.stringify({message:'Someone changed my account email without my permission.',request_id:crypto.randomUUID()})});
 const response=await handle(request,env,deps),body=await response.json();assert.equal(response.status,200);assert.equal(body.mode,'live');assert.equal(body.needs_human,true);assert.ok(body.evidence.length);assert.equal(called,1);
});
test('same-origin, input size and absent storage configuration are enforced',async()=>{
 const req=(body,origin='https://hiver.test')=>new Request('https://hiver.test/api/support',{method:'POST',headers:{origin,'content-type':'application/json'},body});
 assert.equal((await handle(req('{}','https://attacker.test'),{})).status,403);
 assert.equal((await handle(req('x'.repeat(24001)),{})).status,413);
 assert.equal((await handle(new Request('https://hiver.test/api/health'),{})).status,503);
 assert.throws(()=>parseInput({message:'hi'}));
 const hits=retrieve('My Android audio pauses',documents(''));assert.ok(hits.some(x=>x.id==='KB-01'));
 assert.throws(()=>validateOutput({...valid,cited_doc_ids:['fake']},hits));assert.ok(riskSignals('I was charged twice').length);
});
