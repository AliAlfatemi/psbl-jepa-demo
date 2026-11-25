/* PSBL-JEPA demo.
   full mode   : onnxruntime-web runs the frozen encoders in the browser; linear and kNN read-outs follow the paper's
                 frozen-evaluation protocol for the epoch-30, seed-0 checkpoints.
   static mode : no model artifacts are shipped; results for the 12 clips were precomputed offline with the same code.
   Every run carries a generation token, so a result is only ever drawn against the clip it was computed for. */
/* global ort */

const CFG = window.DEMO_CONFIG || { mode: "full", version: "dev" };
const FULL = CFG.mode === "full";
const MODELS = ["vicreg", "psbl"];
const LABEL = { vicreg: "VICReg", psbl: "PSBL (ours)" };
const CHANNELS = ["concentration", "velocity x", "velocity y",
  "orientation D xx", "orientation D xy", "orientation D yx", "orientation D yy",
  "strain-rate E xx", "strain-rate E xy", "strain-rate E yx", "strain-rate E yy"];
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];
const DIV_NEG = ["#104281", "#2a78d6", "#86b6ef"], DIV_POS = ["#f3b1a5", "#e34948", "#9c1f1f"];
const CLIP_CACHE = 2, LOAD_TIMEOUT_MS = 90000;
const TEST_DELAY_MS = Math.min(20000, +new URLSearchParams(location.search).get("testDelay") || 0);   // used only by the race test

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const state = {
  clips: null, manifestSel: null, clipIdx: 0, channel: 0, model: "both", colorBy: "alpha", reveal: false, knn: false,
  revealed: new Set(), cache: new Map(), raw: new Map(), frames: null, probe: {}, map: {}, results: null, pre: null,
  models: Object.fromEntries(MODELS.map((m) => [m, { status: "idle", session: null, error: null, knn: "idle" }])),
  runGeneration: 0, running: false, fallback: false, live: {}, hover: null, ortReady: null, worker: null, wq: new Map(), wid: 0,
};

const isDark = () => document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 2) => (v >= 0 ? "" : "−") + Math.abs(v).toFixed(d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function setStatus(t) { $("#status").textContent = t; }

/* ---------------------------------------------------------------- colour */
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function lerpStops(stops, t) {
  t = Math.min(1, Math.max(0, t)); const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
const rgb = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
const seqStops = () => (isDark() ? [...SEQ].reverse().slice(1) : SEQ.slice(1));
const seqColor = (t) => lerpStops(seqStops(), t);
const divStops = () => [...DIV_NEG, isDark() ? "#383835" : "#f0efec", ...DIV_POS];
const divColor = (t) => lerpStops(divStops(), (t + 1) / 2);

/* ---------------------------------------------------------------- fetch helpers */
async function fetchWithTimeout(url, onProgress) {
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), LOAD_TIMEOUT_MS);
  try {
    const r = await fetch(url, { signal: ctl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${url.split("/").slice(-2).join("/")}`);
    const total = +r.headers.get("content-length") || 0;
    if (!onProgress || !r.body) return new Uint8Array(await r.arrayBuffer());
    const reader = r.body.getReader(), parts = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onProgress(got, total); }
    const out = new Uint8Array(got); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  } catch (e) {
    throw e.name === "AbortError" ? new Error(`timed out after ${LOAD_TIMEOUT_MS / 1000} s loading ${url.split("/").pop()}`) : e;
  } finally { clearTimeout(timer); }
}
const getJSON = async (u) => JSON.parse(new TextDecoder().decode(await fetchWithTimeout(u)));
const mb = (b) => `${(b / 1e6).toFixed(1)} MB`;

/* ---------------------------------------------------------------- runtime (full mode, lazy) */
function wasmSupported() { try { return typeof WebAssembly === "object" && WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])); } catch { return false; } }
function loadOrt() {
  if (!state.ortReady) {
    state.ortReady = new Promise((res, rej) => {
      if (!wasmSupported()) { rej(new Error("WebAssembly is not supported by this browser")); return; }
      const s = document.createElement("script"); s.src = "vendor/ort.wasm.min.js";
      s.onload = () => {
        ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
        const threads = self.crossOriginIsolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;
        ort.env.wasm.numThreads = threads;
        $("#pill-runtime").textContent = `WebAssembly · ${threads} thread${threads > 1 ? "s" : ""}${self.crossOriginIsolated ? "" : " (isolation unavailable)"}`;
        res();
      };
      s.onerror = () => rej(new Error("could not load the ONNX runtime"));
      document.head.appendChild(s);
    });
    state.ortReady.catch(() => { state.ortReady = null; });
  }
  return state.ortReady;
}
async function ensureSession(m, onStage) {
  const ms = state.models[m];
  if (ms.session) return ms.session;
  ms.status = "loading"; renderResults();
  try {
    onStage(`loading runtime (11 MB, once)`);
    await loadOrt();
    const bytes = await fetchWithTimeout(`assets/${m}/encoder.onnx`, (g, t) => onStage(`downloading ${LABEL[m]} weights ${mb(g)}${t ? ` / ${mb(t)}` : ""}`));
    onStage(`initialising ${LABEL[m]}`);
    if (TEST_DELAY_MS) await new Promise((r) => setTimeout(r, TEST_DELAY_MS));
    ms.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    if (!state.probe[m]) state.probe[m] = await getJSON(`assets/${m}/probe.json`);
    ms.status = "ready"; ms.error = null;
    return ms.session;
  } catch (e) { ms.status = "error"; ms.error = e.message; ms.session = null; throw e; }
}
function worker() {
  if (!state.worker) {
    state.worker = new Worker("knn-worker.js");
    state.worker.onmessage = (ev) => { const cb = state.wq.get(ev.data.id ?? `load-${ev.data.model}`); if (cb) { state.wq.delete(ev.data.id ?? `load-${ev.data.model}`); cb(ev.data); } };
  }
  return state.worker;
}
async function ensureKnn(m, onStage) {
  const ms = state.models[m];
  if (ms.knn === "ready") return;
  ms.knn = "loading";
  try {
    const f = await fetchWithTimeout(`assets/${m}/train_feats.f16`, (g, t) => onStage(`downloading ${LABEL[m]} training features ${mb(g)}${t ? ` / ${mb(t)}` : ""}`));
    const l = await fetchWithTimeout(`assets/${m}/train_labels.f32`);
    const p = state.probe[m] || (state.probe[m] = await getJSON(`assets/${m}/probe.json`));
    if (f.byteLength !== p.n_train * p.feature_dim * 2 || l.byteLength !== p.n_train * 8) throw new Error("training-feature shape mismatch");
    await new Promise((res) => { state.wq.set(`load-${m}`, res); worker().postMessage({ type: "load", model: m, feats: f.buffer, labels: l.buffer, d: p.feature_dim }, [f.buffer, l.buffer]); });
    ms.knn = "ready";
  } catch (e) { ms.knn = "error"; throw e; }
}
function knnQuery(m, xs, k) {
  const id = ++state.wid;
  return new Promise((res) => { state.wq.set(id, (d) => res(d.knn)); worker().postMessage({ type: "query", id, model: m, xs, k }); });
}

/* ---------------------------------------------------------------- read-outs */
function standardize(m, z) {
  const p = state.probe[m], xs = new Float32Array(p.feature_dim);
  for (let j = 0; j < p.feature_dim; j++) xs[j] = (z[j] - p.x_mean[j]) / p.x_std[j];
  return xs;
}
function linearReadout(m, xs) {
  const p = state.probe[m];
  return p.W.map((w, r) => w.reduce((s, wj, j) => s + wj * xs[j], p.b[r]) * p.y_std[r] + p.y_mean[r]);
}
function project(m, xs) { const mp = state.map[m]; return mp.components.map((c) => c.reduce((s, cj, j) => s + cj * (xs[j] - mp.mean[j]), 0)); }

/* ---------------------------------------------------------------- clips */
function lruGet(map, k) { if (!map.has(k)) return undefined; const v = map.get(k); map.delete(k); map.set(k, v); return v; }
function lruSet(map, k, v) { map.set(k, v); while (map.size > CLIP_CACHE) map.delete(map.keys().next().value); }
async function clipRaw(i) {          // 8-bit clip as stored (2.2 MB); fetched only for inference or non-preview channels
  let u8 = lruGet(state.raw, i);
  if (!u8) { u8 = await fetchWithTimeout(`assets/clips/${state.clips[i].file}`); lruSet(state.raw, i, u8); }
  return u8;
}
async function clipTensor(i) {       // float32 model input, rebuilt from the 8-bit clip
  let x = lruGet(state.cache, i);
  if (x) return x;
  const c = state.clips[i], u8 = await clipRaw(i), [C, T, H, W] = c.shape, per = T * H * W;
  x = new Float32Array(u8.length);
  for (let ch = 0; ch < C; ch++) { const lo = c.lo[ch], sc = c.scale[ch], o = ch * per; for (let j = 0; j < per; j++) x[o + j] = lo + u8[o + j] * sc; }
  lruSet(state.cache, i, x);
  return x;
}
const labelsVisible = (i) => state.reveal || state.revealed.has(i);
function buildClipGrid() {
  const g = $("#clip-grid"); g.innerHTML = "";
  state.clips.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "clip"; b.dataset.idx = i; b.disabled = state.running; b.setAttribute("aria-pressed", i === state.clipIdx);
    const lab = labelsVisible(i) ? `α ${fmt(c.labels.alpha, 0)} · ζ ${fmt(c.labels.zeta, 0)}` : "";
    b.setAttribute("aria-label", `Clip ${i + 1}${lab ? `, ${lab}` : ""}`);
    b.innerHTML = `<img src="assets/clips/thumb_${String(i).padStart(2, "0")}.png" alt="" loading="lazy"><span>Clip ${i + 1}${lab ? ` · ${lab}` : ""}</span>`;
    b.addEventListener("click", () => selectClip(i));
    g.appendChild(b);
  });
}
function invalidate() { state.runGeneration += 1; state.live = {}; }
async function selectClip(i) {
  if (state.running) return;
  state.clipIdx = i; invalidate();
  $$(".clip").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.idx === i));
  $("#frames-clip").textContent = `· clip ${i + 1}`;
  renderResults(); drawMaps();
  await drawFrames();
}
function frameCanvases(idx, ch, T, W, H) {
  const wrap = $("#frames"); wrap.innerHTML = "";
  return Array.from({ length: T }, (_, t) => {
    const fig = document.createElement("figure"), cv = document.createElement("canvas");
    cv.width = W; cv.height = H; cv.setAttribute("role", "img");
    cv.setAttribute("aria-label", `Clip ${idx + 1}, ${CHANNELS[ch]}, context frame ${t + 1}`);
    fig.appendChild(cv);
    const cap = document.createElement("figcaption"); cap.textContent = `t = ${t + 1}`; fig.appendChild(cap);
    wrap.appendChild(fig);
    return cv;
  });
}
function colorbar(ch, l, h, diverging) {
  const stops = (diverging ? divStops() : seqStops()).join(",");
  $("#colorbar").innerHTML = `<span>${l.toPrecision(3)}</span><span class="bar" style="background:linear-gradient(90deg,${stops})"></span><span>${h.toPrecision(3)}</span>
    <span class="muted">${CHANNELS[ch]} · ${diverging ? "diverging, zero-centred" : "sequential"} · scaled per clip and channel, so colours are not comparable across clips</span>`;
}
async function drawFrames() {
  const idx = state.clipIdx, ch = state.channel, c = state.clips[idx], [, T, H, W] = c.shape;
  if (ch === 0 && !state.raw.has(idx) && state.frames) {
    // pre-rendered preview of the concentration channel (no 2.2 MB download)
    const img = new Image(); img.src = `assets/clips/frames_${String(idx).padStart(2, "0")}_${isDark() ? "dark" : "light"}.png`;
    try { await img.decode(); } catch { setStatus(`Could not load the preview for clip ${idx + 1}.`); return; }
    if (idx !== state.clipIdx || ch !== state.channel) return;
    frameCanvases(idx, ch, T, W, H).forEach((cv, t) => cv.getContext("2d").drawImage(img, t * W, 0, W, H, 0, 0, W, H));
    const f = state.frames[idx]; colorbar(ch, f.lo, f.hi, false);
    return;
  }
  let u8;
  try { u8 = await clipRaw(idx); } catch (e) { setStatus(`Could not load clip ${idx + 1}: ${e.message}`); return; }
  if (idx !== state.clipIdx || ch !== state.channel) return;
  const per = T * H * W, o = ch * per, lo = c.lo[ch], sc = c.scale[ch];
  let qmin = 255, qmax = 0;
  for (let j = 0; j < per; j++) { const q = u8[o + j]; if (q < qmin) qmin = q; if (q > qmax) qmax = q; }
  const vlo = lo + qmin * sc, vhi = lo + qmax * sc, diverging = ch !== 0, amax = Math.max(Math.abs(vlo), Math.abs(vhi)) || 1;
  const lut = new Uint8ClampedArray(256 * 3);   // colour per stored 8-bit level: 256 evaluations instead of 200 k
  for (let q = 0; q < 256; q++) {
    const col = diverging ? divColor((lo + q * sc) / amax) : seqColor((q - qmin) / ((qmax - qmin) || 1));
    lut[3 * q] = col[0]; lut[3 * q + 1] = col[1]; lut[3 * q + 2] = col[2];
  }
  frameCanvases(idx, ch, T, W, H).forEach((cv, t) => {
    const ctx = cv.getContext("2d"), img = ctx.createImageData(W, H), d = img.data, base = o + t * H * W;
    for (let p = 0; p < H * W; p++) { const q = 3 * u8[base + p]; d[4 * p] = lut[q]; d[4 * p + 1] = lut[q + 1]; d[4 * p + 2] = lut[q + 2]; d[4 * p + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  });
  if (diverging) colorbar(ch, -amax, amax, true); else colorbar(ch, vlo, vhi, false);
}

/* ---------------------------------------------------------------- run */
function setBusy(b) {
  state.running = b;
  $("#live").setAttribute("aria-busy", b);
  $("#run-btn").disabled = b; $("#cancel-btn").hidden = !b;
  $$(".clip, .seg-btn[data-model], #channel-select, #knn-toggle, #reveal").forEach((el) => { el.disabled = b; });
}
async function run(only) {
  state.runGeneration += 1;
  const generation = state.runGeneration, clipIdx = state.clipIdx, idx = clipIdx, wantKnn = state.knn;
  const models = only ? [only] : state.model === "both" ? MODELS : [state.model];
  if (!only) state.live = {};
  setBusy(true); renderResults();
  // Before committing any result: drop it if a newer run, a cancel, or a clip/model change happened meanwhile.
  const alive = () => generation === state.runGeneration && clipIdx === state.clipIdx;
  try {
    const c = state.clips[idx];
    if (!FULL || state.fallback) {
      for (const m of models) state.live[m] = { ...state.pre.clips[idx].models[m], source: "precomputed" };
      state.revealed.add(idx); buildClipGrid(); renderResults(); drawMaps();
      setStatus(`Showing precomputed results for clip ${idx + 1}.`);
      return;
    }
    const x = await clipTensor(idx);
    for (const m of models) {
      if (!alive()) return;
      try {
        const s = await ensureSession(m, (t) => alive() && setStatus(t));
        if (!alive()) return;
        setStatus(`Running ${LABEL[m]} encoder in your browser…`);
        const t0 = performance.now();
        const out = await s.run({ context: new ort.Tensor("float32", x, [1, ...c.shape]) });
        const msEnc = performance.now() - t0;
        if (!alive()) return;
        const z = out.embedding.data, ref = c.torch[m].embedding_quantized_input;
        let parity = 0; for (let j = 0; j < z.length; j++) parity = Math.max(parity, Math.abs(z[j] - ref[j]));
        const xs = standardize(m, z), t1 = performance.now();
        const lin = linearReadout(m, xs), xy = project(m, xs), msLin = performance.now() - t1;
        state.live[m] = { linear: lin, xy, msEnc, msLin, parity, source: "live", knn: null };
        state.revealed.add(idx); buildClipGrid(); renderResults(); drawMaps();
        if (wantKnn) {
          await ensureKnn(m, (t) => alive() && setStatus(t));
          if (!alive()) return;
          setStatus(`Searching ${state.probe[m].n_train.toLocaleString()} training windows (${LABEL[m]})…`);
          const t2 = performance.now(), pz = state.probe[m], kz = await knnQuery(m, xs, pz.knn_k);
          const knn = kz.map((v, r) => v * pz.y_std[r] + pz.y_mean[r]);   // back to physical units
          if (!alive()) return;
          Object.assign(state.live[m], { knn, msKnn: performance.now() - t2 });
          renderResults();
        }
      } catch (e) {
        console.error(e);
        if (!alive()) return;
        state.live[m] = { error: e.message, source: "error" };
        renderResults();
      }
    }
    if (alive()) {
      const errs = models.filter((m) => state.live[m] && state.live[m].error);
      setStatus(errs.length ? `Finished with ${errs.length} error${errs.length > 1 ? "s" : ""}; see the card${errs.length > 1 ? "s" : ""} below.` : "Done. Every number above was computed on this device.");
    }
  } finally {
    if (generation === state.runGeneration) setBusy(false);
  }
}
function cancel() {
  if (!state.running) return;
  state.runGeneration += 1; setBusy(false); setStatus("Cancelled. Results that arrive late are discarded.");
}

/* ---------------------------------------------------------------- result cards */
function renderResults() {
  const idx = state.clipIdx, c = state.clips[idx], wrap = $("#results");
  const show = labelsVisible(idx), truth = [c.labels.alpha, c.labels.zeta];
  wrap.innerHTML = "";
  for (const m of MODELS) {
    const r = state.live[m], ms = state.models[m], card = document.createElement("div");
    card.className = "result-card";
    const row = (name, v) => `<span>${name}</span><span>${v ? fmt(v[0]) : "—"}</span><span>${v ? fmt(v[1]) : "—"}</span>` +
      `<span class="err">${v && show ? `${fmt(Math.abs(v[0] - truth[0]))} · ${fmt(Math.abs(v[1] - truth[1]))}` : ""}</span>`;
    let body;
    if (r && r.error) {
      body = `<p class="error">Could not run ${LABEL[m]}: ${esc(r.error)}</p><div class="run-row"><button type="button" class="retry" data-retry="${m}">Retry ${LABEL[m]}</button>
        <button type="button" class="retry" data-pre="${m}">Show precomputed result</button></div>`;
    } else {
      const knnLabel = `kNN (k=${state.probe[m] ? state.probe[m].knn_k : (state.pre ? state.pre.k[m] : "–")})`;
      const knnVal = r && r.knn ? r.knn : null;
      body = `<div class="kv"><span class="h"></span><span class="h">α</span><span class="h">ζ</span><span class="h">|error| α · ζ</span>
        <span>true</span><span>${show ? fmt(truth[0]) : "hidden"}</span><span>${show ? fmt(truth[1]) : "hidden"}</span><span></span>
        ${row("linear probe", r && r.linear)}
        ${row(knnLabel, knnVal)}</div>`;
      const meta = [];
      if (r && r.source === "live") {
        meta.push(`encoder ${(r.msEnc / 1000).toFixed(2)} s · linear ${r.msLin.toFixed(1)} ms${r.msKnn != null ? ` · kNN ${r.msKnn.toFixed(0)} ms` : ""}`);
        meta.push(`runtime parity: max |ONNX − PyTorch| on the 8-bit input ${r.parity.toExponential(1)}`);
      }
      if (r && r.source === "precomputed") meta.push("precomputed offline with the same checkpoint and code; nothing ran in your browser");
      if (r) meta.push(`quantization impact: max |PyTorch(8-bit) − PyTorch(full precision)| ${c.quant[m].toExponential(1)}`);
      if (r && r.source === "live" && !r.knn) meta.push(state.knn ? "kNN pending…" : "kNN off: tick “Include kNN read-out” to load the training features");
      if (!r) meta.push(ms.status === "loading" ? "loading…" : "not run for this clip yet");
      body += `<p class="meta">${meta.join("<br>")}</p>`;
    }
    card.innerHTML = `<h3><span class="swatch s-${m}" aria-hidden="true"></span>${LABEL[m]}</h3>${body}`;
    wrap.appendChild(card);
  }
  $$("[data-pre]").forEach((b) => b.addEventListener("click", () => {
    const m = b.dataset.pre; state.live[m] = { ...state.pre.clips[state.clipIdx].models[m], source: "precomputed" };
    state.revealed.add(state.clipIdx); buildClipGrid(); renderResults(); drawMaps();
  }));
  $$("[data-retry]").forEach((b) => b.addEventListener("click", () => {
    const m = b.dataset.retry; state.models[m] = { status: "idle", session: null, error: null, knn: state.models[m].knn === "ready" ? "ready" : "idle" };
    run(m);
  }));
}

/* ---------------------------------------------------------------- maps */
const mapGeom = {};
function drawMaps() {
  if (!state.map.psbl) return;
  const key = state.colorBy === "alpha" ? 0 : 1;
  const all = MODELS.flatMap((m) => state.map[m].labels.map((l) => l[key]));
  const lo = Math.min(...all), hi = Math.max(...all), traj = state.clips[state.clipIdx].trajectory;
  const PAL = Array.from({ length: 64 }, (_, i) => rgb(seqColor(i / 63)));   // one palette per draw, not per point
  const colorOf = (v) => PAL[Math.round((v - lo) / ((hi - lo) || 1) * 63)];
  const showTraj = labelsVisible(state.clipIdx);
  for (const m of MODELS) {
    const cv = $(`.map[data-model="${m}"] canvas`), dpr = devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight; if (!W) continue;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
    const mp = state.map[m], pts = mp.xy, live = state.live[m] && state.live[m].xy ? state.live[m] : null;
    const xs = pts.map((p) => p[0]).concat(live ? [live.xy[0]] : []), ys = pts.map((p) => p[1]).concat(live ? [live.xy[1]] : []);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)], pl = 26, pr = 12, pt = 12, pb = 24;
    const sx = (v) => pl + (v - x0) / ((x1 - x0) || 1) * (W - pl - pr), sy = (v) => H - pb - (v - y0) / ((y1 - y0) || 1) * (H - pt - pb);
    mapGeom[m] = { sx, sy, pts, mp };
    ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.fillText("PC1 →", W / 2, H - 6);
    ctx.save(); ctx.translate(11, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("PC2 →", 0, 0); ctx.restore();
    const surf = css("--surface");
    pts.forEach((p, i) => {
      if (showTraj && mp.traj[i] === traj) return;
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 2.6, 0, 2 * Math.PI);
      ctx.fillStyle = colorOf(mp.labels[i][key]); ctx.fill();
    });
    if (showTraj) {
      ctx.lineWidth = 1.5; ctx.strokeStyle = css("--text-2");
      pts.forEach((p, i) => { if (mp.traj[i] !== traj) return; ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 3.4, 0, 2 * Math.PI); ctx.fillStyle = surf; ctx.fill(); ctx.stroke(); });
    }
    if (live) {
      const X = sx(live.xy[0]), Y = sy(live.xy[1]);
      ctx.beginPath(); ctx.arc(X, Y, 8, 0, 2 * Math.PI); ctx.lineWidth = 4; ctx.strokeStyle = surf; ctx.stroke();
      ctx.lineWidth = 2.5; ctx.strokeStyle = css("--ring"); ctx.stroke();
    }
    if (state.hover && state.hover.m === m) {
      const p = pts[state.hover.i]; ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 5, 0, 2 * Math.PI); ctx.lineWidth = 2; ctx.strokeStyle = css("--ring"); ctx.stroke();
    }
  }
  const stops = Array.from({ length: 8 }, (_, i) => rgb(seqColor(i / 7))).join(",");
  $("#ramp-legend").innerHTML = `<span>${state.colorBy === "alpha" ? "α" : "ζ"} ${fmt(lo, 0)}</span><span class="bar" style="background:linear-gradient(90deg,${stops})"></span><span>${fmt(hi, 0)}</span>`;
}
function spearman(a, b) {
  const rank = (v) => { const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]), r = new Array(v.length); let i = 0;
    while (i < o.length) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1; } return r; };
  const ra = rank(a), rb = rank(b), n = a.length, ma = ra.reduce((s, x) => s + x, 0) / n, mb2 = rb.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0; for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb2); da += (ra[i] - ma) ** 2; db += (rb[i] - mb2) ** 2; }
  return num / Math.sqrt(da * db);
}
function mapSummaries() {
  let rows = "";
  for (const m of MODELS) {
    const mp = state.map[m], pc1 = mp.xy.map((p) => p[0]), pc2 = mp.xy.map((p) => p[1]);
    const a = mp.labels.map((l) => l[0]), z = mp.labels.map((l) => l[1]);
    const r = [[spearman(pc1, a), spearman(pc2, a)], [spearman(pc1, z), spearman(pc2, z)]];
    $(`[data-summary="${m}"]`).textContent = `Summary: PC1 explains ${(mp.explained[0] * 100).toFixed(0)}% and PC2 ${(mp.explained[1] * 100).toFixed(0)}% of variance. ` +
      `Rank correlation with α: PC1 ${fmt(r[0][0])}, PC2 ${fmt(r[0][1])}; with ζ: PC1 ${fmt(r[1][0])}, PC2 ${fmt(r[1][1])}.`;
    $(`.map[data-model="${m}"] canvas`).setAttribute("aria-label", `${LABEL[m]} validation embedding map: 1,776 points. ${$(`[data-summary="${m}"]`).textContent}`);
  }
  const n = Math.max(...state.map.psbl.traj) + 1, byTraj = {};
  for (const m of MODELS) { byTraj[m] = Array.from({ length: n }, () => []); state.map[m].traj.forEach((t, i) => byTraj[m][t].push(i)); }
  for (let t = 0; t < n; t++) {
    const cells = MODELS.map((m) => { const mp = state.map[m], idx = byTraj[m][t];
      const c = [0, 1].map((d) => idx.reduce((s, i) => s + mp.xy[i][d], 0) / idx.length); return `<td>${fmt(c[0])}</td><td>${fmt(c[1])}</td>`; });
    const l = state.map.psbl.labels[state.map.psbl.traj.indexOf(t)];
    rows += `<tr><td>${t + 1}</td><td>${fmt(l[0], 0)}</td><td>${fmt(l[1], 0)}</td>${cells.join("")}</tr>`;
  }
  $("#map-table").innerHTML = `<div class="table-wrap" tabindex="0"><table><caption class="sr-only">Centroid of each validation trajectory in each map</caption><thead><tr><th>Trajectory</th><th>α</th><th>ζ</th><th>VICReg PC1</th><th>VICReg PC2</th><th>PSBL PC1</th><th>PSBL PC2</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function bindMapPointer() {
  const tip = $("#tooltip"), panel = $(".map-panel");
  $$(".map canvas").forEach((cv) => {
    const m = cv.closest(".map").dataset.model;
    const show = (e) => {
      const g = mapGeom[m]; if (!g) return;
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let bi = -1, bd = e.pointerType === "touch" ? 400 : 144;
      g.pts.forEach((p, i) => { const d = (g.sx(p[0]) - mx) ** 2 + (g.sy(p[1]) - my) ** 2; if (d < bd) { bd = d; bi = i; } });
      if (bi < 0) { tip.hidden = true; state.hover = null; drawMaps(); return; }
      state.hover = { m, i: bi }; drawMaps();
      const l = g.mp.labels[bi], pr = panel.getBoundingClientRect(), xy = g.pts[bi];
      tip.innerHTML = `<strong>${LABEL[m]}</strong><br>α ${fmt(l[0], 0)} · ζ ${fmt(l[1], 0)}<br>PC1 ${fmt(xy[0])} · PC2 ${fmt(xy[1])}<br><span class="muted">trajectory ${g.mp.traj[bi] + 1} of 24</span>`;
      tip.hidden = false;
      let left = e.clientX - pr.left + 14; const top = e.clientY - pr.top + 14;
      if (left + tip.offsetWidth > pr.width - 8) left = e.clientX - pr.left - tip.offsetWidth - 14;
      tip.style.left = `${Math.max(4, left)}px`; tip.style.top = `${top}px`;
    };
    cv.addEventListener("pointermove", show);
    cv.addEventListener("pointerdown", show);
    cv.addEventListener("pointerleave", (e) => { if (e.pointerType !== "touch") { tip.hidden = true; state.hover = null; drawMaps(); } });
  });
}

/* ---------------------------------------------------------------- charts */
function drawCharts() {
  const R = state.results.curves, wrap = $("#charts"); wrap.innerHTML = "";
  const specs = [
    { key: "effective_rank", title: "Effective rank (of 128)", log: false },
    { key: "std_mean", title: "Pooled embedding std (log scale)", log: true },
    { key: "participation_ratio", title: "Participation ratio", log: false },
    { key: "psbl_ref_latent_spearman", title: "PSBL: latent vs. reference distance (Spearman)", log: false, only: ["psbl"] },
  ];
  for (const sp of specs) {
    const W = 320, H = 190, L = 38, Rr = 10, T = 10, B = 26, models = sp.only || MODELS;
    const series = models.map((m) => {
      const runs = R[m], ep = runs[0].epoch, vals = ep.map((_, t) => runs.map((r) => r[sp.key][t]).filter((v) => v != null));
      return { m, ep, mean: vals.map((v) => v.reduce((a, b) => a + b, 0) / v.length), min: vals.map((v) => Math.min(...v)), max: vals.map((v) => Math.max(...v)) };
    });
    const tf = sp.log ? Math.log10 : (v) => v, allv = series.flatMap((s) => [...s.min, ...s.max]).map(tf);
    let y0 = Math.min(...allv), y1 = Math.max(...allv); const padv = (y1 - y0) * 0.08; y0 -= padv; y1 += padv;
    if (!sp.log && y0 < 0 && Math.min(...series.flatMap((s) => s.min)) >= 0) y0 = 0;
    const X = (e) => L + (e - 1) / 29 * (W - L - Rr), Y = (v) => T + (1 - (tf(v) - y0) / (y1 - y0)) * (H - T - B);
    const ticks = sp.log ? [-3, -2, -1].filter((t) => t >= y0 && t <= y1).map((t) => [10 ** t, `10${["⁻³", "⁻²", "⁻¹"][t + 3]}`]) : niceTicks(y0, y1).map((t) => [t, String(+t.toFixed(2))]);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${sp.title}, epochs 1 to 30, ${models.map((m) => LABEL[m]).join(" and ")}; data table below">`;
    svg += `<g class="axis">${ticks.map(([v, lab]) => `<line x1="${L}" x2="${W - Rr}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 3}" text-anchor="end">${lab}</text>`).join("")}`;
    svg += [1, 10, 20, 30].map((e) => `<text x="${X(e)}" y="${H - 12}" text-anchor="middle">${e}</text>`).join("") + `<text x="${(L + W) / 2}" y="${H - 1}" text-anchor="middle">epoch</text></g>`;
    for (const s of series) {
      const col = `var(--${s.m})`;
      const band = s.ep.map((e, t) => `${X(e)},${Y(s.max[t])}`).concat(s.ep.map((e, t) => `${X(e)},${Y(s.min[t])}`).reverse()).join(" ");
      svg += `<polygon points="${band}" fill="${col}" opacity=".16"/><polyline points="${s.ep.map((e, t) => `${X(e)},${Y(s.mean[t])}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    }
    svg += `<line class="hover-line" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/><rect x="${L}" y="${T}" width="${W - L - Rr}" height="${H - T - B}" fill="transparent"/></svg>`;
    const rows = series[0].ep.map((e, t) => `<tr><td>${e}</td>${series.map((s) => `<td>${fmtv(s.mean[t], sp.log)}</td><td>${fmtv(s.min[t], sp.log)}–${fmtv(s.max[t], sp.log)}</td>`).join("")}</tr>`).join("");
    const div = document.createElement("div"); div.className = "chart";
    div.innerHTML = `<h3>${sp.title}</h3>${svg}<div class="tooltip" hidden></div>
      <details class="data-table"><summary>Show data table</summary><div class="table-wrap" tabindex="0"><table><thead><tr><th>Epoch</th>${series.map((s) => `<th>${LABEL[s.m]} mean</th><th>range</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div></details>`;
    wrap.appendChild(div);
    const svgEl = div.querySelector("svg"), hl = div.querySelector(".hover-line"), tip = div.querySelector(".tooltip");
    const move = (e) => {
      const r = svgEl.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
      const ep = Math.min(30, Math.max(1, Math.round((px - L) / (W - L - Rr) * 29 + 1)));
      hl.setAttribute("x1", X(ep)); hl.setAttribute("x2", X(ep)); hl.setAttribute("visibility", "visible");
      tip.innerHTML = `<strong>epoch ${ep}</strong>` + series.map((s) => `<br><span class="swatch s-${s.m}"></span>${LABEL[s.m]} ${fmtv(s.mean[ep - 1], sp.log)}`).join("");
      tip.hidden = false;
      const dr = div.getBoundingClientRect(); let left = e.clientX - dr.left + 12;
      if (left + tip.offsetWidth > dr.width) left = e.clientX - dr.left - tip.offsetWidth - 12;
      tip.style.left = `${Math.max(4, left)}px`; tip.style.top = `${e.clientY - dr.top + 12}px`;
    };
    svgEl.addEventListener("pointermove", move); svgEl.addEventListener("pointerdown", move);
    svgEl.addEventListener("pointerleave", () => { hl.setAttribute("visibility", "hidden"); tip.hidden = true; });
  }
}
const fmtv = (v, log) => (log ? v.toExponential(2) : v.toFixed(v < 10 ? 3 : 1));
function niceTicks(a, b) {
  const step = Math.pow(10, Math.floor(Math.log10((b - a) / 3))), m = [1, 2, 5, 10].find((k) => (b - a) / (k * step) <= 4) * step;
  const out = []; for (let t = Math.ceil(a / m) * m; t <= b + 1e-9; t += m) out.push(+t.toFixed(6)); return out;
}

/* ---------------------------------------------------------------- results table */
function drawTable() {
  const R = state.results, ci = R.ci, d = R.dct_val;
  const pm = (v) => `${v[0].toFixed(3)}<span class="muted"> ± ${v[1].toFixed(3)}</span>`;
  let h = `<caption class="sr-only">Frozen-encoder MSE by split, epoch and encoder</caption><thead><tr><th>Split · epoch</th><th>Encoder</th>
    <th class="det">Linear α</th><th class="det">Linear ζ</th><th>Linear mean</th><th class="det">kNN α</th><th class="det">kNN ζ</th><th>kNN mean</th></tr></thead><tbody>`;
  for (const r of R.table) {
    h += `<tr class="${r.split === "test" ? "test" : ""}"><td>${r.split === "test" ? "test" : "validation"} · ${r.epoch}</td><td><span class="swatch s-${r.method}" aria-hidden="true"></span>${LABEL[r.method]}</td>
      <td class="det">${pm(r.lin_alpha)}</td><td class="det">${pm(r.lin_zeta)}</td><td>${pm(r.linear)}</td><td class="det">${pm(r.knn_alpha)}</td><td class="det">${pm(r.knn_zeta)}</td><td>${pm(r.knn)}</td></tr>`;
  }
  h += `<tr><td>validation</td><td>fixed DCT descriptor (no learning)</td><td class="det">${d.lin_alpha.toFixed(3)}</td><td class="det">${d.lin_zeta.toFixed(3)}</td><td>${d.linear.toFixed(3)}</td>
    <td class="det">${d.knn_alpha.toFixed(3)}</td><td class="det">${d.knn_zeta.toFixed(3)}</td><td>${d.knn.toFixed(3)}</td></tr></tbody>`;
  $("#res-table").innerHTML = h;
  $("#pm").textContent = R.predict_mean_val.toFixed(3);
  $("#ci-text").textContent = `No clear overall separation with three seeds. VICReg has the lower mean linear test MSE (${R.table.find((r) => r.split === "test" && r.method === "vicreg").linear[0].toFixed(3)} vs ` +
    `${R.table.find((r) => r.split === "test" && r.method === "psbl").linear[0].toFixed(3)}); test kNN is nearly tied. Paired PSBL − VICReg differences: linear ${fmt(ci.linear[0], 3)} ` +
    `(95% CI ${fmt(ci.linear[1], 3)} to ${fmt(ci.linear[2], 3)}), kNN ${fmt(ci.knn[0], 3)} (95% CI ${fmt(ci.knn[1], 3)} to ${fmt(ci.knn[2], 3)}). Both intervals include zero; they are not equivalence tests.`;
}

/* ---------------------------------------------------------------- text that depends on the mode */
function modeText() {
  $("#lede").innerHTML = FULL
    ? `Self-supervised JEPA encoders trained on The Well <code>active_matter</code> simulations, running <strong>live in your browser</strong>. Pick a held-out validation clip and a frozen encoder estimates the activity α and alignment ζ of the physics, with no server involved.`
    : `Self-supervised JEPA encoders trained on The Well <code>active_matter</code> simulations. Explore held-out validation clips, what frozen encoders predict for them, and the paper's aggregate findings.`;
  $("#live-lede").textContent = FULL
    ? "Each clip is four context frames (11 physical channels, 224×224) from a validation trajectory. The encoder maps it to a 128-number embedding; a single linear layer and, optionally, a k-nearest-neighbour lookup over the training set turn that embedding into α and ζ, using the same frozen-evaluation protocol as the paper, with the epoch-30 seed-0 checkpoint shown for interactive illustration."
    : "Each clip is four context frames (11 physical channels, 224×224) from a validation trajectory. Predictions were precomputed offline from the epoch-30 seed-0 checkpoint with the same frozen-evaluation protocol as the paper; this public version ships no model weights or training data.";
  if (FULL) {
    const b = $("#private-banner"); b.hidden = false; b.textContent = "PRIVATE COURSE DEMO — DO NOT REDISTRIBUTE";
  }
  if (!FULL) {
    const n = $("#mode-notice"); n.hidden = false;
    n.textContent = "Public static version: results for the 12 clips are precomputed. The interactive in-browser model is not publicly released while the project is under course evaluation.";
    $("#badge").textContent = "Precomputed validation illustration · epoch 30 · seed 0";
    $("#run-btn").textContent = "Reveal precomputed result";
    $("#knn-toggle").closest("label").hidden = true;
    $("#pill-runtime").textContent = "precomputed results";
  }
  $("#how-demo").textContent = FULL
    ? "How this demo works: the encoder runs in WebAssembly via onnxruntime-web. Inputs are validation windows stored at 8-bit precision per channel; embeddings are standardized with training-split statistics and read out by a linear head refit with the paper's validation-selected settings, and by distance-weighted kNN over standardized training features (loaded only if you enable it). Each result card reports runtime parity (ONNX vs. PyTorch on the same 8-bit input) and the separate quantization impact (8-bit vs. full-precision input, both in PyTorch)."
    : "How this page works: for each clip the PyTorch encoder output on the 8-bit stored input was read out offline (linear head and kNN, as in the paper's protocol). The page draws those stored values; no model, training feature, label or probe parameter is shipped.";
  $("#third-party").innerHTML = FULL
    ? "<strong>Third-party code:</strong> onnxruntime-web 1.20.1 (MIT, Microsoft) and coi-serviceworker 0.1.7 (MIT); licenses in <code>vendor/</code>. No analytics, fonts or other third-party requests."
    : "<strong>Third-party code:</strong> none. No analytics, fonts or other third-party requests.";
  $("#version-line").textContent = `Demo version: ${CFG.version}${CFG.built ? ` (${CFG.built})` : ""} · Models: epoch 30, seed 0 · Paper results: three seeds · Mode: ${FULL ? "private full demo" : "public static demo"}`;
  $("#footer-precision").textContent = FULL
    ? "Runtime parity (ONNX in the browser vs. PyTorch, same 8-bit input) stayed below 10⁻⁶ on all 12 clips in our automated tests; every live run reports its own value. Quantization impact (8-bit vs. full-precision input, PyTorch) is at most 4.1·10⁻⁵ across the 12 clips."
    : "Precomputed values come from the PyTorch encoder on the 8-bit stored inputs; 8-bit storage changes the embedding by at most 4.1·10⁻⁵ across the 12 clips.";
}

/* ---------------------------------------------------------------- wiring */
function bindControls() {
  const sel = $("#channel-select");
  CHANNELS.forEach((n, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${i} · ${n}`; sel.appendChild(o); });
  sel.addEventListener("change", () => { state.channel = +sel.value; drawFrames(); });
  $$(".seg-btn[data-model]").forEach((b) => b.addEventListener("click", () => {
    if (state.running) return;
    state.model = b.dataset.model; invalidate();
    $$(".seg-btn[data-model]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    renderResults(); drawMaps();
  }));
  $$(".color-btn").forEach((b) => b.addEventListener("click", () => {
    state.colorBy = b.dataset.color; $$(".color-btn").forEach((x) => x.setAttribute("aria-pressed", x === b)); drawMaps();
  }));
  $("#reveal").addEventListener("change", (e) => { state.reveal = e.target.checked; buildClipGrid(); renderResults(); drawMaps(); });
  $("#knn-toggle").addEventListener("change", (e) => { state.knn = e.target.checked; });
  $("#run-btn").addEventListener("click", () => run());
  $("#cancel-btn").addEventListener("click", cancel);
  addEventListener("resize", () => drawMaps());
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawMaps(); drawFrames(); });
}

window.__demoState = state; // read-only hook used by tests/browser_test.mjs

(async function main() {
  modeText(); bindControls();
  try {
    const [clips, results, pre, frames] = await Promise.all([getJSON("assets/clips/clips.json"), getJSON("assets/results.json"), getJSON("assets/precomputed.json"),
      getJSON("assets/clips/frames.json").catch(() => null)]);
    state.frames = frames;
    state.clips = clips.clips.map((c) => ({ ...c, quant: Object.fromEntries(MODELS.map((m) => [m, c.torch[m].max_abs_diff_quantization])) }));
    state.results = results; state.pre = pre;
    buildClipGrid(); renderResults(); drawCharts(); drawTable();
    $("#frames-clip").textContent = "· clip 1";
    await drawFrames();
  } catch (e) { console.error(e); setStatus(`Could not load the demo data: ${e.message}`); return; }
  // maps are optional: a failure here must not disable the rest of the page
  await Promise.all(MODELS.map(async (m) => {
    try { state.map[m] = await getJSON(`assets/${m}/val_map.json`); $(`[data-ev="${m}"]`).textContent = `· PC1 ${(state.map[m].explained[0] * 100).toFixed(0)}% · PC2 ${(state.map[m].explained[1] * 100).toFixed(0)}%`; }
    catch (e) { $(`[data-summary="${m}"]`).textContent = `Map unavailable: ${e.message}`; }
  }));
  if (state.map.psbl && state.map.vicreg) { drawMaps(); bindMapPointer(); (window.requestIdleCallback || setTimeout)(() => mapSummaries()); }
  $("#run-btn").disabled = false;
  if (FULL && !wasmSupported()) {
    state.fallback = true;
    $("#pill-runtime").textContent = "WebAssembly unavailable · precomputed fallback";
    $("#run-btn").textContent = "Reveal precomputed result";
    setStatus("This browser cannot run WebAssembly, so results for the 12 clips are shown from offline precomputation.");
  } else {
    setStatus(FULL ? "Ready. The encoder (≈ 8 MB) downloads when you press Run." : "Ready. Pick a clip and press Reveal precomputed result.");
    if (FULL) $("#pill-runtime").textContent = `WebAssembly${self.crossOriginIsolated ? " · multi-threaded" : " · single-threaded"}`;
  }
})();
