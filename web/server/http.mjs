import {createHmac,randomUUID} from 'node:crypto';
import {VERSION,ServiceError,parseInput,documents,retrieve,generate,finalDecision} from './core.mjs';
import {reserveUsage,storageReady} from './usage.mjs';
function json(value,status=200,extra={}){return new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff',...extra}});}
async function readJSON(request){
 if(!request.headers.get('content-type')?.toLowerCase().startsWith('application/json'))throw new ServiceError('content_type','Send JSON input.',415);
 const reader=request.body?.getReader();if(!reader)throw new ServiceError('invalid_input','Enter a message.');
 const chunks=[];let size=0;while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>24000){await reader.cancel();throw new ServiceError('input_too_large','This request is too large.',413);}chunks.push(value);}
 const bytes=new Uint8Array(size);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length;}
 try{return JSON.parse(new TextDecoder().decode(bytes));}catch{throw new ServiceError('invalid_json','The request could not be read.');}
}
export async function handle(request,env=process.env,deps={reserve:reserveUsage,ready:storageReady,generate}){
 const url=new URL(request.url),route=url.searchParams.get('route')||url.pathname.split('/').at(-1);
 if(route==='health'){
  let ready=!!(env.GEMINI_API_KEY&&env.BLOB_READ_WRITE_TOKEN&&env.RATE_LIMIT_SECRET);
  if(ready)try{await deps.ready();}catch{ready=false;}
  return json({status:ready?'ready':'not_configured',version:VERSION,mode:'live',hosting:'vercel',retention:'No messages, policies, or replies are persisted on the server.'},ready?200:503);
 }
 if(route==='knowledge'&&request.method==='GET')return json({documents:documents('')});
 if(route!=='support')return json({error:{code:'not_found',message:'Endpoint not found.'}},404);
 const id=randomUUID(),started=Date.now();
 try{
  if(request.method!=='POST')return json({error:{code:'method_not_allowed',message:'Use the support form.'}},405,{Allow:'POST'});
  if(request.headers.get('origin')!==url.origin)throw new ServiceError('origin_denied','Use the form on this website.',403);
  const raw=await readJSON(request),input=parseInput(raw);
  if(typeof raw.request_id!=='string'||!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(raw.request_id))throw new ServiceError('invalid_request_id','Reload the page and try again.');
  if(!env.GEMINI_API_KEY||!env.BLOB_READ_WRITE_TOKEN||!env.RATE_LIMIT_SECRET)throw new ServiceError('not_configured','The live service is being configured. Please use the recorded evaluation for now.',503);
  const ip=request.headers.get('x-vercel-forwarded-for')||request.headers.get('x-real-ip')||'shared-unknown-client';
  const hash=createHmac('sha256',env.RATE_LIMIT_SECRET).update(ip).digest('hex').slice(0,32);
  await deps.reserve(hash,raw.request_id);
  const retrievalStart=Date.now(),evidence=retrieve(input.message,documents(input.policy),input.history),retrievalMs=Date.now()-retrievalStart;
  const generationStart=Date.now(),generated=await deps.generate(input,evidence,env),generationMs=Date.now()-generationStart;
  const decision=finalDecision(input,evidence,generated.result);
  console.log(JSON.stringify({event:'support_completed',request_id:id,latency_ms:Date.now()-started,routing:decision.routing,version:VERSION}));
  return json({request_id:id,version:VERSION,mode:'live',model:generated.model,...decision,evidence:evidence.map(d=>({...d,cited:decision.cited_doc_ids.includes(d.id)})),timing:{retrieval_ms:retrievalMs,generation_ms:generationMs,total_ms:Date.now()-started},usage:generated.usage,privacy:{redacted:input.message!==raw.message.trim(),stored_on_server:false},knowledge_source:input.policy?'visitor_supplied_policy':'demo_music_policy'});
 }catch(error){
  const known=error instanceof ServiceError,status=known?error.status:503;
  console.log(JSON.stringify({event:'support_failed',request_id:id,status,code:known?error.code:'service_unavailable',latency_ms:Date.now()-started}));
  return json({request_id:id,error:{code:known?error.code:'service_unavailable',message:known?error.message:'The support service is unavailable. No reply was sent. Please try again later.'}},status,known&&error.retryAfter?{'Retry-After':String(error.retryAfter)}:{});
 }
}
