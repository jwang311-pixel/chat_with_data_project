const modelSelect = document.getElementById('model');
const modeSelect = document.getElementById('mode');
const datasetPreview = document.getElementById('datasetPreview');
const questionEl = document.getElementById('question');
const askBtn = document.getElementById('askBtn');
const uploadInput = document.getElementById('upload');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const chatStatus = document.getElementById('chatStatus');
const finalAnswer = document.getElementById('finalAnswer');
const executionInfo = document.getElementById('executionInfo');
const pythonCode = document.getElementById('pythonCode');

const DATASETS_ENDPOINT = '/api/datasets';
const MODELS_ENDPOINT = '/api/models';
const PROMPT_MODES_ENDPOINT = '/api/prompt-modes';
const UPLOAD_ENDPOINT = '/api/upload';
const CHAT_ENDPOINT = '/api/chat';

let datasetCache = [];
let currentDatasetId = null;
let currentDatasetPreview = [];
let currentDatasetMeta = null;

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`Failed to load ${path}${text ? `: ${text}` : ''}`);
  }
  return await response.json();
}

function asArray(data, keys = []) {
  if (Array.isArray(data)) return data;
  for (const key of keys) {
    if (data && Array.isArray(data[key])) return data[key];
  }
  return [];
}

function fillSelect(select, items, valueKey = 'id', labelKey = 'label') {
  select.innerHTML = '';
  if (!items || items.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = '(none)';
    select.appendChild(option);
    return;
  }

  for (const item of items) {
    const option = document.createElement('option');
    option.value = item?.[valueKey] ?? '';
    option.textContent = item?.[labelKey] ?? '(unnamed)';
    select.appendChild(option);
  }
}

function clearPreview(message = 'Upload a CSV or Excel file to see preview.') {
  datasetPreview.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'small';
  div.textContent = message;
  datasetPreview.appendChild(div);
}

function renderPreviewTable(rows) {
  datasetPreview.innerHTML = '';

  if (!Array.isArray(rows) || rows.length === 0) {
    clearPreview('No preview rows available.');
    return;
  }

  const columns = [...new Set(rows.flatMap(row => Object.keys(row || {})))];

  const shell = document.createElement('div');
  shell.style.overflow = 'auto';
  shell.style.border = '1px solid rgba(255,255,255,0.12)';
  shell.style.borderRadius = '10px';
  shell.style.marginTop = '10px';

  const table = document.createElement('table');
  table.style.width = '100%';
  table.style.borderCollapse = 'collapse';
  table.style.minWidth = '520px';
  table.style.background = '#0b1220';

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');

  columns.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col;
    th.style.position = 'sticky';
    th.style.top = '0';
    th.style.background = '#111827';
    th.style.zIndex = '1';
    th.style.fontWeight = '700';
    th.style.textAlign = 'left';
    th.style.borderBottom = '1px solid rgba(255,255,255,0.12)';
    th.style.borderRight = '1px solid rgba(255,255,255,0.12)';
    th.style.padding = '8px 10px';
    th.style.fontSize = '13px';
    th.style.whiteSpace = 'nowrap';
    headRow.appendChild(th);
  });

  thead.appendChild(headRow);

  const tbody = document.createElement('tbody');
  rows.forEach(row => {
    const tr = document.createElement('tr');

    columns.forEach(col => {
      const td = document.createElement('td');
      const value = row?.[col];
      td.textContent = value === null || value === undefined ? '' : String(value);
      td.style.borderBottom = '1px solid rgba(255,255,255,0.12)';
      td.style.borderRight = '1px solid rgba(255,255,255,0.12)';
      td.style.padding = '8px 10px';
      td.style.fontSize = '13px';
      td.style.whiteSpace = 'nowrap';
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  shell.appendChild(table);
  datasetPreview.appendChild(shell);
}

function setCurrentDataset(dataset) {
  if (!dataset) {
    currentDatasetId = null;
    currentDatasetMeta = null;
    currentDatasetPreview = [];
    clearPreview();
    return;
  }

  currentDatasetId = dataset.dataset_id ?? dataset.id ?? null;
  currentDatasetMeta = dataset;
  currentDatasetPreview = dataset.preview || dataset.preview_rows || [];
  renderPreviewTable(currentDatasetPreview);
}

async function refreshDatasets() {
  const raw = await loadJson(DATASETS_ENDPOINT);
  const datasets = asArray(raw, ['datasets', 'items', 'data']);
  datasetCache = datasets;

  if (datasets.length === 0) {
    currentDatasetId = null;
    currentDatasetMeta = null;
    currentDatasetPreview = [];
    clearPreview();
    return;
  }

  const stillExists = currentDatasetId
    && datasets.some(ds => (ds.dataset_id ?? ds.id ?? '') === currentDatasetId);

  if (!stillExists) {
    setCurrentDataset(datasets[0]);
  } else {
    const ds = datasets.find(x => (x.dataset_id ?? x.id ?? '') === currentDatasetId);
    if (ds) {
      currentDatasetMeta = ds;
      currentDatasetPreview = ds.preview || ds.preview_rows || [];
      renderPreviewTable(currentDatasetPreview);
    }
  }
}

async function refreshModels() {
  const raw = await loadJson(MODELS_ENDPOINT);
  const models = asArray(raw, ['models', 'items', 'data']);
  fillSelect(modelSelect, models, 'id', 'label');
}

async function refreshModes() {
  const raw = await loadJson(PROMPT_MODES_ENDPOINT);
  const modes = asArray(raw, ['modes', 'items', 'data']);
  fillSelect(modeSelect, modes, 'id', 'label');
}

uploadBtn.addEventListener('click', async () => {
  const file = uploadInput.files?.[0];
  if (!file) {
    uploadStatus.textContent = 'Please choose a CSV or Excel file first.';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  uploadStatus.textContent = 'Uploading...';
  uploadBtn.disabled = true;

  try {
    const response = await fetch(UPLOAD_ENDPOINT, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();
    uploadStatus.innerHTML = `<span class="ok">Uploaded:</span> ${data.display_name ?? data.filename ?? data.dataset_id ?? 'file'}`;

    setCurrentDataset(data);

    await refreshDatasets();
  } catch (error) {
    uploadStatus.innerHTML = `<span class="err">Upload failed:</span> ${error.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
});

askBtn.addEventListener('click', async () => {
  const question = questionEl.value.trim();
  if (!question) {
    chatStatus.textContent = 'Please type a question first.';
    return;
  }

  if (!currentDatasetId) {
    if (datasetCache.length > 0) {
      setCurrentDataset(datasetCache[0]);
    } else {
      chatStatus.textContent = 'Please upload a dataset first.';
      return;
    }
  }

  chatStatus.textContent = 'Thinking...';
  finalAnswer.textContent = 'Working...';
  executionInfo.textContent = '';
  pythonCode.textContent = '';
  askBtn.disabled = true;

  try {
    const response = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: currentDatasetId,
        question,
        model_id: modelSelect.value,
        prompt_mode: modeSelect.value,
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();

    finalAnswer.textContent = data.final_answer ?? '(empty)';
    executionInfo.textContent =
      `result:\n${data.execution_result ?? '(none)'}\n\n` +
      `stdout:\n${data.execution_stdout || '(none)'}\n\n` +
      `error:\n${data.execution_error || '(none)'}`;
    pythonCode.textContent = data.python_code || '(no code returned)';
    chatStatus.innerHTML = `<span class="ok">Done.</span> Used ${data.trace_count ?? 1} trace(s).`;
  } catch (error) {
    chatStatus.innerHTML = `<span class="err">Request failed:</span> ${error.message}`;
    finalAnswer.textContent = 'Request failed.';
  } finally {
    askBtn.disabled = false;
  }
});

(async function init() {
  try {
    await Promise.all([
      refreshDatasets(),
      refreshModels(),
      refreshModes(),
    ]);

    if (!currentDatasetId && datasetCache.length > 0) {
      setCurrentDataset(datasetCache[0]);
    }

    chatStatus.textContent = 'Ready.';
  } catch (error) {
    chatStatus.innerHTML = `<span class="err">Startup error:</span> ${error.message}`;
    clearPreview('Failed to load preview.');
  }
})();