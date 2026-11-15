'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const title = value => String(value ?? 'Unavailable').replaceAll('_',' ');
const pct = value => `${(value*100).toFixed(1)}%`;
let data, selected, reviews = new Map();
const find = id => data.cases.find(c => c.id === id);
const disagrees = c => c.judge && c.human && c.judge.acceptable !== c.human.acceptable;

function filtered() {
  const query = $('search').value.toLowerCase().trim();
  const mode = $('filter').value;
  return data.cases.filter(c => (!query || `${c.text} ${c.intent} ${c.id}`.toLowerCase().includes(query)) && (
    mode==='all' || mode==='missed' && c.expected_escalation && !c.escalated ||
    mode==='escalated' && c.escalated || mode==='wrong' && c.intent!==c.expected_intent || mode==='disagreement' && disagrees(c)));
}

function renderLibrary() {
  const cases = filtered();
  $('case-count').textContent = `${cases.length} cases`;
  $('case-list').replaceChildren();
  if (!cases.length) { $('case-list').innerHTML = '<p class="empty">No matching cases. Try another search or filter.</p>'; return; }
  for (const c of cases) {
    const button = document.createElement('button');
    button.className = `case-item${selected===c.id?' active':''}`;
    button.setAttribute('aria-pressed', String(selected===c.id));
    button.dataset.id = c.id;
    const missed = c.expected_escalation && !c.escalated;
    button.innerHTML = `<span class="intent-name">${escapeHTML(title(c.intent))}</span><span class="case-text">${escapeHTML(c.text)}</span><span class="case-meta"><span>#${c.id}</span><span class="${missed?'warning':''}">${missed?'Missed escalation':c.escalated?'Human needed':'Auto-handled'}</span></span>`;
    button.addEventListener('click', () => selectCase(c.id));
    $('case-list').append(button);
  }
}

function renderAssessment(c) {
  const render = (label, rating) => !rating ? `<div><h4>${label}</h4><p>No valid ${label==='LLM judge'?'judge':'human'} rating is available for this case.</p></div>` : `<div><h4>${label}</h4><p><strong>${rating.acceptable?'Accepted':'Rejected'}</strong></p>${['grounded','addresses_ask','no_overpromise','tone_ok'].map(k=>`<p>${rating[k]?'✓':'×'} ${escapeHTML(title(k))}</p>`).join('')}${rating.note?`<p>${escapeHTML(rating.note)}</p>`:''}</div>`;
  $('assessment').innerHTML = `<div class="assessment-grid">${render('LLM judge',c.judge)}${render('Human rater',c.human)}</div><p class="muted small">Acceptability requires grounding, addressing the ask, and no overpromising. Tone is scored separately.</p>${c.judge?.addresses_ask_rationale?`<p class="small"><strong>Judge’s relevance rationale:</strong> ${escapeHTML(c.judge.addresses_ask_rationale)}</p>`:''}`;
}

function selectCase(id, scroll=false) {
  const c = find(id);
  if (!c) return;
  selected = c.id;
  $('case-id').textContent = `CASE #${c.id} / TEST SPLIT`;
  $('message').textContent = c.text;
  $('intent').textContent = title(c.intent);
  $('intent-truth').textContent = `Human label: ${title(c.expected_intent)} · ${c.intent===c.expected_intent?'match':'mismatch'}`;
  $('routing').textContent = c.escalated ? 'Hand over to a human' : 'Auto-handle';
  $('routing-reason').textContent = c.reason ? `${title(c.reason)} · ${c.source || 'content rule'}` : `Confidence ${c.confidence.toFixed(2)} · Top similarity ${c.similarity.toFixed(2)}`;
  const missed = c.expected_escalation && !c.escalated;
  const falseAlarm = !c.expected_escalation && c.escalated;
  $('failure').hidden = !missed && !falseAlarm;
  $('failure').textContent = missed ? `Missed escalation. The human label calls for review (${title(c.expected_reason)}), but the original policy auto-handled this message. A high confidence score did not catch that risk.` : falseAlarm ? 'Extra escalation. The human label did not require review, but the original policy requested it. This is a false positive under the labelled policy.' : '';
  $('reply').textContent = c.reply;
  $('draft-status').textContent = 'AI DRAFT · UNSENT';
  $('evidence-count').textContent = `${c.evidence.length} cases · ${c.evidence.filter(e=>e.cited).length} cited`;
  $('evidence').innerHTML = c.evidence.map(e=>`<div class="evidence-item${e.cited?' cited':''}"><div class="evidence-head"><span>EXAMPLE [${e.position}] · #${e.id}</span>${e.cited?'<b>● CITED BY DRAFTER</b>':'<span>RETRIEVED</span>'}</div><p>${escapeHTML(e.customer)}</p><p><strong>Historical reply:</strong> ${escapeHTML(e.reply)}</p></div>`).join('');
  renderAssessment(c);
  $('review-reply').value = reviews.get(id)?.reply ?? c.reply;
  $('review-status').textContent = reviews.get(id)?.status ?? 'Awaiting your review. Edits are local to this session.';
  renderLibrary();
  const url = new URL(location.href); url.searchParams.set('case',c.id); history.replaceState(null,'',url);
  if (scroll) $('workspace').scrollIntoView({behavior:'smooth'});
}

function renderBaselines() {
  $('baseline-chart').innerHTML = data.baselines.map(b=>`<div class="baseline-row"><div class="baseline-label"><span>${escapeHTML(b.name)}</span><span>${pct(b.accuracy)}</span></div><div class="baseline-track"><div class="baseline-fill" style="width:${b.accuracy*100}%"></div></div><div class="baseline-f1">Macro F1 ${b.macro_f1.toFixed(2)} · ${b.correct}/${b.n} intent labels correct</div></div>`).join('');
}

function renderPolicy() {
  const tau = Number($('confidence').value), sigma=Number($('similarity').value);
  $('confidence-value').textContent=tau.toFixed(2); $('similarity-value').textContent=sigma.toFixed(2);
  let tp=0, fp=0, fn=0, count=0;
  for (const c of data.cases) {
    const review=c.escalated || c.confidence<tau || c.similarity<sigma;
    if (review) count++;
    if (review && c.expected_escalation) tp++;
    if (review && !c.expected_escalation) fp++;
    if (!review && c.expected_escalation) fn++;
  }
  $('policy-result').innerHTML = `<div><strong>${count} / 90</strong><span>sent to human review</span></div><div><strong>${pct(tp/(tp+fn))}</strong><span>escalation recall · ${tp}/7 caught</span></div><div><strong>${count?pct(tp/(tp+fp)):'—'}</strong><span>escalation precision</span></div><div><strong>${fp}</strong><span>extra escalations vs. labels</span></div>`;
}

function saveReview(status) {
  reviews.set(selected,{id:selected,reply:$('review-reply').value,status});
  $('review-status').textContent=status;
}

async function init() {
  const response=await fetch('data.json');
  if (!response.ok) throw new Error('Data unavailable');
  data=await response.json();
  renderBaselines(); renderPolicy();
  $('provenance').textContent = `Original evaluation · source ${data.source_commit.slice(0,10)} · 90 test records`;
  const initial=new URLSearchParams(location.search).get('case');
  selectCase(find(initial)?initial:'166674');
  for (const id of ['search','filter']) $(id).addEventListener(id==='search'?'input':'change',()=>{renderLibrary();const matches=filtered();if(matches.length && !matches.some(c=>c.id===selected)) selectCase(matches[0].id);});
  document.querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>{ $('filter').value='all'; $('search').value=''; selectCase(b.dataset.case,true); }));
  document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{ $('filter').value=b.dataset.filter; $('search').value=''; const c=filtered()[0]; if(c)selectCase(c.id,true); else renderLibrary(); }));
  $('review-reply').addEventListener('input',()=>saveReview('Edited draft · not yet reviewed. Nothing has been sent.'));
  $('approve').addEventListener('click',()=>saveReview('Marked reviewed in this session. Nothing has been sent.'));
  $('hold').addEventListener('click',()=>saveReview('Held for a human in this session. Nothing has been sent.'));
  $('reset-review').addEventListener('click',()=>{reviews.delete(selected);selectCase(selected);});
  $('confidence').addEventListener('input',renderPolicy); $('similarity').addEventListener('input',renderPolicy);
  $('reset-thresholds').addEventListener('click',()=>{$('confidence').value=data.tau;$('similarity').value=data.sigma;renderPolicy();});
  $('export-review').addEventListener('click',()=>{
    const blob=new Blob([JSON.stringify({notice:'Visitor reviews from the public replay, not original benchmark annotations.',source_commit:data.source_commit,reviews:[...reviews.values()]},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob), a=document.createElement('a');a.href=url;a.download='hiver-session-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
}
init().catch(error=>{$('load-error').hidden=false;console.error(error);});
