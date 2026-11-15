import {get,put} from '@vercel/blob';
import {ServiceError} from './core.mjs';
const PATH='hiver/usage-v1.json';
export function mutateUsage(previous,ip,id,now=Math.floor(Date.now()/1000)){
 const state={version:1,counters:{...(previous?.counters||{})},requests:{...(previous?.requests||{})}};
 for(const [key,value] of Object.entries(state.counters))if(value.expires<=now)delete state.counters[key];
 for(const [key,expires] of Object.entries(state.requests))if(expires<=now)delete state.requests[key];
 const requestKey=`${ip}:${id}`;
 if(state.requests[requestKey])throw new ServiceError('duplicate_request','This request was already submitted. Start a new request for another answer.',409);
 const minute=Math.floor(now/60),day=Math.floor(now/86400);
 const buckets=[[`ip-minute:${minute}:${ip}`,5,(minute+2)*60],[`ip-day:${day}:${ip}`,30,(day+2)*86400],[`global-minute:${minute}`,15,(minute+2)*60],[`global-day:${day}`,200,(day+2)*86400]];
 for(const [key,limit] of buckets)if((state.counters[key]?.used||0)>=limit)throw new ServiceError('usage_limit','The demo’s request allowance has been reached. Please try again later.',429,60);
 for(const [key,,expires] of buckets)state.counters[key]={used:(state.counters[key]?.used||0)+1,expires};
 state.requests[requestKey]=now+86400;
 return state;
}
async function readState(){
 try{const result=await get(PATH,{access:'private',useCache:false});
  if(!result)return {state:null,etag:null};
  if(!result.stream||!result.blob.etag)throw new Error('Missing storage version');
  const text=await new Response(result.stream).text();
  return {state:JSON.parse(text),etag:result.blob.etag};
 }catch(error){if(error.name==='BlobNotFoundError')return {state:null,etag:null};throw error;}
}
export async function reserveUsage(ip,id,io={read:readState,write:put},now=Math.floor(Date.now()/1000)){
 for(let attempt=0;attempt<5;attempt++){
  const {state,etag}=await io.read();
  const next=mutateUsage(state,ip,id,now);
  try{await io.write(PATH,JSON.stringify(next),{access:'private',addRandomSuffix:false,allowOverwrite:!!etag,...(etag?{ifMatch:etag}:{}),contentType:'application/json'});return;}
  catch(error){if(!['BlobPreconditionFailedError','BlobAlreadyExistsError'].includes(error.name))throw error;}
 }
 throw new ServiceError('service_busy','The service is handling several requests. Please try again shortly.',503,5);
}
export async function storageReady(){await readState();return true;}
