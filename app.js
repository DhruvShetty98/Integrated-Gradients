const input = document.querySelector('#imageInput');
const dropzone = document.querySelector('#dropzone');
const analyze = document.querySelector('#analyzeButton');
const results = document.querySelector('#results');
const loading = document.querySelector('#loading');
let chosenFile = null;
let response = null;

function selectFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  chosenFile = file;
  analyze.disabled = false;
  dropzone.querySelector('strong').textContent = file.name;
  dropzone.querySelector('span').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to analyze`;
}
input.addEventListener('change', () => selectFile(input.files[0]));
['dragenter','dragover'].forEach(type => dropzone.addEventListener(type, e => {e.preventDefault(); dropzone.classList.add('drag');}));
['dragleave','drop'].forEach(type => dropzone.addEventListener(type, e => {e.preventDefault(); dropzone.classList.remove('drag');}));
dropzone.addEventListener('drop', e => selectFile(e.dataTransfer.files[0]));

function imageUrl(file) { return URL.createObjectURL(file); }
function baselineName(value) { return ({neutral:'Neutral / dataset mean',black:'Black image',white:'White image',blurred:'Blurred image'})[value]; }
async function run(targetIndex = null) {
  if (!chosenFile) return;
  loading.classList.remove('hidden');
  try {
    const form = new FormData();
    form.append('image', chosenFile);
    if (targetIndex !== null) form.append('target_index', targetIndex);
    const res = await fetch('/api/predict', {method:'POST', body:form});
    if (!res.ok) throw new Error((await res.json()).detail || 'Analysis failed');
    response = await res.json(); render();
  } catch (err) { alert(err.message); }
  finally { loading.classList.add('hidden'); }
}
function render() {
  document.querySelector('#inputPreview').src = imageUrl(chosenFile);
  document.querySelector('#primaryLabel').textContent = response.prediction.label;
  document.querySelector('#primaryConfidence').textContent = `${response.prediction.confidence.toFixed(2)}% confidence`;
  const maps = document.querySelector('#explanationMaps'); maps.innerHTML = '';
  const differences = document.querySelector('#differenceMaps'); differences.innerHTML = '';
  response.explanations.forEach(item => {
    const map = document.createElement('article'); map.className = 'map-card';
    map.innerHTML = `<div><p class="card-label">${baselineName(item.baseline)}</p><span>Δ ${item.convergence_delta}</span></div><div class="map-stack"><figure><img src="${item.baseline_preview}" alt="${baselineName(item.baseline)} reference input" /><figcaption>baseline reference</figcaption></figure><figure><img src="${item.overlay}" alt="${baselineName(item.baseline)} attribution overlay" /><figcaption>model input + overlay</figcaption></figure><figure><img src="${item.heatmap}" alt="${baselineName(item.baseline)} Integrated Gradients heatmap" /><figcaption>IG evidence strength</figcaption></figure></div>`;
    maps.appendChild(map);
    const difference = document.createElement('article'); difference.className = 'map-card';
    if (item.baseline === 'neutral') {
      difference.innerHTML = `<div><p class="card-label">Neutral / dataset mean</p><span>reference</span></div><div class="reference-empty">Reference<br>map</div>`;
    } else {
      difference.innerHTML = `<div><p class="card-label">${baselineName(item.baseline)}</p><span>mean shift ${item.mean_difference}</span></div><img src="${item.comparison_overlay}" alt="Change from neutral for ${baselineName(item.baseline)}" />`;
    }
    differences.appendChild(difference);
  });
  document.querySelector('#metrics').textContent = `EACH MAP USES 48 IG STEPS · SMALLER CONVERGENCE Δ IS PREFERRED`;
  const list = document.querySelector('#predictionList'); list.innerHTML = '';
  response.predictions.forEach(item => {
    const el = document.createElement('button');
    el.className = `prediction ${item.index === response.prediction.index ? 'active' : ''}`;
    el.innerHTML = `<div><b>${item.label}</b><span class="bar"><i style="width:${item.confidence}%"></i></span></div><span>${item.confidence.toFixed(1)}%</span>`;
    el.onclick = () => run(item.index); list.appendChild(el);
  });
  results.classList.remove('hidden'); results.scrollIntoView({behavior:'smooth', block:'start'});
}
analyze.addEventListener('click', () => run());
document.querySelector('#newImage').addEventListener('click', () => { window.scrollTo({top:0,behavior:'smooth'}); input.click(); });
