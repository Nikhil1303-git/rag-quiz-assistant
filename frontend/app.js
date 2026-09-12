const $ = (id) => document.getElementById(id);
const setBusy = (button, busy, label) => { button.disabled = busy; button.firstChild.textContent = busy ? 'Working...' : label; };

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function renderStatus(status) {
  $('api-health').textContent = 'online';
  $('store-health').textContent = status.collection_info?.error ? 'not ready' : 'ready';
  $('store-health').className = `badge ${status.collection_info?.error ? '' : 'badge-live'}`;
  $('document-count').textContent = status.collection_info?.document_count ?? '0';
  $('provider').textContent = status.llm_provider || '--';
  $('model').textContent = status.llm_model || '--';
  $('embedding').textContent = status.collection_info?.embedding_model || '--';
  $('chunk-size').textContent = `${status.chunk_size ?? '--'} chars`;
  $('connection-label').textContent = 'Pipeline connected';
  $('connection-label').previousElementSibling.classList.add('live');
}

async function refreshStatus() {
  try { renderStatus(await api('/api/status')); }
  catch (error) { $('api-health').textContent = 'offline'; $('api-health').className = 'badge'; $('connection-label').textContent = error.message; }
}

$('pdf-file').addEventListener('change', (event) => { $('file-label').textContent = event.target.files[0]?.name || 'Choose a PDF'; });
['dragenter', 'dragover'].forEach((eventName) => $('dropzone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropzone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => $('dropzone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropzone').classList.remove('dragging'); }));
$('dropzone').addEventListener('drop', (event) => { if (event.dataTransfer.files.length) { $('pdf-file').files = event.dataTransfer.files; $('file-label').textContent = event.dataTransfer.files[0].name; } });

$('upload-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = $('pdf-file').files[0]; const result = $('ingest-result'); const button = event.target.querySelector('button');
  if (!file) { result.textContent = 'Choose a PDF file first.'; result.className = 'result error'; return; }
  setBusy(button, true, 'Ingest document'); result.textContent = 'Extracting, chunking, and embedding...'; result.className = 'result';
  const data = new FormData(); data.append('file', file); data.append('recreate', $('recreate').checked);
  try { const body = await api('/api/ingest', { method:'POST', body:data }); result.textContent = `Indexed ${body.chunks_created} chunks from ${body.pages_extracted} pages.`; result.className = 'result success'; await refreshStatus(); }
  catch (error) { result.textContent = error.message; result.className = 'result error'; } finally { setBusy(button, false, 'Ingest document'); }
});

$('query-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = event.target.querySelector('button'); const answer = $('answer'); const sources = $('sources');
  setBusy(button, true, 'Ask the index'); answer.className = 'answer loading'; answer.textContent = 'Searching the index and composing an answer...'; sources.innerHTML = '';
  try { const body = await api('/api/query', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ query:$('question').value, top_k:Number($('top-k').value) }) }); answer.className = 'answer'; answer.textContent = body.response; sources.innerHTML = body.retrieved_documents.map((doc, index) => `<div class="source"><strong>Retrieved chunk ${index + 1}</strong>${doc}</div>`).join(''); }
  catch (error) { answer.className = 'answer error'; answer.textContent = error.message; } finally { setBusy(button, false, 'Ask the index'); }
});

let quizQuestions = [];

function renderQuiz() {
  const list = $('quiz-list');
  list.innerHTML = quizQuestions.map((q, index) => `
    <div class="quiz-item" data-id="${q.id}">
      <div class="quiz-question"><strong>Q${index + 1}.</strong> ${q.question}</div>
      <textarea class="quiz-answer" rows="3" placeholder="Type your answer..."></textarea>
      <div class="quiz-actions">
        <button class="ghost-button quiz-submit" type="button">Check answer</button>
      </div>
      <div class="quiz-feedback"></div>
    </div>
  `).join('');

  list.querySelectorAll('.quiz-item').forEach((item) => {
    const id = item.dataset.id;
    const question = quizQuestions.find((q) => q.id === id);
    const button = item.querySelector('.quiz-submit');
    const textarea = item.querySelector('.quiz-answer');
    const feedback = item.querySelector('.quiz-feedback');

    button.addEventListener('click', async () => {
      const userAnswer = textarea.value.trim();
      if (!userAnswer) { feedback.textContent = 'Type an answer first.'; feedback.className = 'quiz-feedback error'; return; }
      button.disabled = true; button.textContent = 'Checking...';
      feedback.textContent = ''; feedback.className = 'quiz-feedback';
      try {
        const result = await api('/api/quiz/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question.question,
            expected_answer: question.expected_answer,
            context: question.context,
            user_answer: userAnswer,
          }),
        });
        feedback.className = `quiz-feedback verdict-${result.verdict.toLowerCase().replace(/\s+/g, '-')}`;
        feedback.innerHTML = `<strong>${result.verdict} (${result.score}/100)</strong><span>${result.feedback}</span>`;
      } catch (error) {
        feedback.textContent = error.message; feedback.className = 'quiz-feedback error';
      } finally {
        button.disabled = false; button.textContent = 'Check answer';
      }
    });
  });
}

$('quiz-generate').addEventListener('click', async () => {
  const button = $('quiz-generate'); const status = $('quiz-status');
  setBusy(button, true, 'Generate quiz');
  status.textContent = 'Reading the indexed document and writing questions...'; status.className = 'result';
  $('quiz-list').innerHTML = '';
  try {
    const body = await api('/api/quiz/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_questions: Number($('quiz-count').value) }),
    });
    quizQuestions = body.questions;
    status.textContent = `Generated ${quizQuestions.length} questions from the indexed document.`;
    status.className = 'result success';
    renderQuiz();
  } catch (error) {
    status.textContent = error.message; status.className = 'result error';
  } finally {
    setBusy(button, false, 'Generate quiz');
  }
});

$('refresh-status').addEventListener('click', refreshStatus);
refreshStatus();
