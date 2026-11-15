'use strict';
(()=>{
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let history=[],queue=[],busy=false,controller=null,activePolicy='';
function queueView(){
 $('queue-count').textContent=`${queue.length} draft${queue.length===1?'':'s'}`;
 $('live-queue').innerHTML=queue.slice(-6).reverse().map(q=>`<div class="queue-item"><strong>${esc(q.result.summary)}</strong><span>${esc(q.status)}</span></div>`).join('');
 $('export-live').disabled=queue.length===0;
}
function inspect(result){
 $('live-inspection').innerHTML=`<span class="route-badge ${result.needs_human?'human':''}">${result.needs_human?'Specialist review required':'Draft ready for review'}</span><div class="inspect-pair"><span>Intent</span><strong>${esc(result.intent.replaceAll('_',' '))}</strong></div><div class="inspect-pair"><span>Model confidence¹</span><strong>${Math.round(result.confidence*100)} / 100</strong></div><div class="inspect-pair"><span>Response time</span><strong>${(result.timing.total_ms/1000).toFixed(1)} seconds</strong></div><div class="inspect-pair"><span>Model</span><strong>${esc(result.model)}</strong></div><div class="inspect-pair"><span>Knowledge</span><strong>${result.knowledge_source==='visitor_supplied_policy'?'Your custom policy':'Demo service policy'}</strong></div><p class="inspect-summary"><strong>Case summary</strong><br>${esc(result.summary)}</p>${result.routing_reasons.length?`<ul class="inspect-reasons">${result.routing_reasons.map(r=>`<li>${esc(r)}</li>`).join('')}</ul>`:'<p class="small muted">No specialist trigger was detected. A person must still review the draft.</p>'}<details open><summary>Retrieved knowledge · ${result.evidence.length}</summary>${result.evidence.map(e=>`<div class="live-source"><strong>${esc(e.id)} · ${esc(e.title)} ${e.cited?'● CITED':''}</strong><p>${esc(e.text)}</p><span class="source-score">BM25 relevance: ${e.score.toFixed(2)} · not a probability</span></div>`).join('')||'<p class="small muted">No relevant knowledge was found. The agent must ask for clarification or hand off.</p>'}</details><p class="small muted" style="font-size:9px;margin-top:14px">¹ Self-reported, uncalibrated. This score is not the chance the answer is correct.</p><p class="tiny" style="font-size:8px;margin-top:10px">TRACE ${esc(result.request_id)}</p>`;
}
function addMessage(role,text){
 const node=document.createElement('article');node.className=`chat-message ${role}`;
 node.innerHTML=`<div class="speaker">${role==='customer'?'CUSTOMER':'HIVER · LIVE AI'}</div><div class="chat-body"></div>`;node.querySelector('.chat-body').textContent=text;$('live-thread').append(node);return node;
}
function addDraft(result){
 const ticket={id:result.request_id,result,reviewed_reply:result.answer,status:result.needs_human?'Awaiting specialist review':'Awaiting draft review'};queue.push(ticket);
 const node=addMessage('assistant',result.answer),body=node.querySelector('.chat-body');
 const chips=document.createElement('div');chips.className='source-chips';chips.innerHTML=result.cited_doc_ids.map(id=>`<span class="source-chip">${esc(id)}</span>`).join('')+`<span class="source-chip">${result.needs_human?'HUMAN HANDOFF':'REVIEW BEFORE USE'}</span>`;body.append(chips);
 const edit=document.createElement('details');edit.className='review-draft';edit.innerHTML='<summary>Review this draft</summary><label class="sr-only">Reviewed draft</label><textarea rows="4" maxlength="2000" aria-label="Reviewed draft"></textarea><div class="live-review-buttons"><button type="button" class="primary" data-action="review">Mark reviewed</button><button type="button" data-action="hold">Hold for specialist</button></div><p class="live-review-status" role="status">Nothing has been sent.</p>';
 const textarea=edit.querySelector('textarea');textarea.value=result.answer;textarea.addEventListener('input',()=>{ticket.reviewed_reply=textarea.value;ticket.status='Edited · awaiting review';queueView();});
 edit.querySelector('[data-action=review]').addEventListener('click',()=>{ticket.reviewed_reply=textarea.value;ticket.status=result.needs_human?'Draft reviewed · specialist handoff still required':'Draft reviewed · not sent';edit.querySelector('.live-review-status').textContent=ticket.status;queueView();});
 edit.querySelector('[data-action=hold]').addEventListener('click',()=>{ticket.reviewed_reply=textarea.value;ticket.status='Held for specialist · not sent';edit.querySelector('.live-review-status').textContent=ticket.status;queueView();});
 body.append(edit);queueView();
}
function resetConversation(){history=[];$('live-thread').innerHTML='<div class="conversation-empty"><span class="empty-mark">h</span><h3>Start a fresh conversation.</h3><p>The next request will not include earlier messages. Your session review queue stays available.</p></div>';$('live-inspection').innerHTML='<h3>Ready for a new case.</h3><p class="small muted">Send a message to inspect the next decision.</p>';$('live-error').hidden=true;}
async function health(){try{const response=await fetch('/api/health',{cache:'no-store'});const value=await response.json();$('live-health').textContent=response.ok&&value.status==='ready'?'● Live AI ready':'Live AI unavailable';$('live-health').classList.toggle('error',!response.ok);}catch{$('live-health').textContent='Live AI unavailable';$('live-health').classList.add('error');}}
$('live-form').addEventListener('submit',async event=>{
 event.preventDefault();if(busy)return;const message=$('live-message').value.trim();if(message.length<4)return;
 const requestedPolicy=$('live-policy').value.trim();if(requestedPolicy!==activePolicy){resetConversation();activePolicy=requestedPolicy;}
 busy=true;$('live-error').hidden=true;$('live-submit').disabled=true;$('live-submit').textContent='Generating…';$('new-conversation').disabled=true;$('live-policy').disabled=true;$('live-style').disabled=true;
 $('live-thread').querySelector('.conversation-empty')?.remove();addMessage('customer',message);
 const progress=document.createElement('div');progress.className='generating';progress.setAttribute('role','status');progress.innerHTML='<span class="dot"></span> Searching support knowledge and generating a reply…';$('live-thread').append(progress);progress.scrollIntoView({block:'nearest',behavior:'smooth'});
 controller=new AbortController();const timeout=setTimeout(()=>controller?.abort(),35000);
 try{
  const response=await fetch('/api/support',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,history,policy:$('live-policy').value,style:$('live-style').value,request_id:crypto.randomUUID()}),signal:controller.signal});
  const result=await response.json();if(!response.ok)throw new Error(result.error?.message||'The support service could not finish this request.');
  progress.remove();addDraft(result);inspect(result);
  history=[...history,{role:'user',text:message},{role:'assistant',text:result.answer}].slice(-4);
  $('live-message').value='';$('live-char-count').textContent='0 / 2,000';$('live-thread').lastElementChild.scrollIntoView({block:'nearest',behavior:'smooth'});
 }catch(error){progress.remove();$('live-error').textContent=error.name==='AbortError'?'The request timed out. No reply was sent. You can try again.':error.message;$('live-error').hidden=false;}
 finally{clearTimeout(timeout);controller=null;busy=false;$('live-submit').disabled=false;$('live-submit').textContent='Generate reply ↗';$('new-conversation').disabled=false;$('live-policy').disabled=false;$('live-style').disabled=false;}
});
$('live-message').addEventListener('input',()=>$('live-char-count').textContent=`${$('live-message').value.length.toLocaleString()} / 2,000`);
document.querySelectorAll('[data-live-sample]').forEach(button=>button.addEventListener('click',()=>{if(busy)return;$('live-message').value=button.dataset.liveSample;$('live-message').dispatchEvent(new Event('input'));$('live-message').focus();}));
$('new-conversation').addEventListener('click',()=>{if(!busy)resetConversation();});
$('live-policy').addEventListener('input',()=>{$('policy-mode').textContent=$('live-policy').value.trim()?'Your custom policy will replace the default. The next request starts fresh context.':'Using 14 demo policy documents.';});
$('view-knowledge').addEventListener('click',async()=>{const panel=$('knowledge-preview');if(!panel.hidden){panel.hidden=true;return;}try{const r=await fetch('/api/knowledge');if(!r.ok)throw new Error();const value=await r.json();panel.innerHTML=value.documents.map(d=>`<h4>${esc(d.id)} · ${esc(d.title)}</h4><p>${esc(d.text)}</p>`).join('');panel.hidden=false;}catch{$('policy-mode').textContent='The demo policies could not be loaded. Please retry.';}});
$('export-live').addEventListener('click',()=>{const file=new Blob([JSON.stringify({version:'live-2.0',exported_at:new Date().toISOString(),notice:'Session-local draft review. No messages were sent and no support team was contacted. Model confidence is not a probability.',tickets:queue},null,2)],{type:'application/json'});const url=URL.createObjectURL(file),a=document.createElement('a');a.href=url;a.download='hiver-live-session-audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
health();
})();
