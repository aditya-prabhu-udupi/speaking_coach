(() => {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const MAX_REC = 60, MIN_SEC = 5, MAX_FILE_SEC = 300, MAX_MB = 25;
const SAVE_KEYS = ["duration_sec", "total_words", "wpm", "filler_count", "filler_per_100", "pause_count", "repeat_count", "unique_ratio", "score"];
const TOPICS = ["Introduce yourself.", "Describe a project you're proud of.", "Explain your favourite hobby.", "Describe your perfect weekend.", "Teach us something you know well.", "Tell us about your last trip."];

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage blocked */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } }
};
const settings = Object.assign({ model: "small", theme: "auto", save: true }, store.get("sc.settings", {}));
let clip = null, busy = false;

const h = (t, a = {}, ...kids) => {
  const e = document.createElement(t);
  for (const [k, v] of Object.entries(a)) k === "class" ? e.className = v : k === "text" ? e.textContent = v : e.setAttribute(k, v);
  e.append(...kids.flat().filter(x => x != null));
  return e;
};
const svg = (t, a = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", t);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  return e;
};
const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/* ---------- alerts ---------- */
function notify(msg) { const a = $("#alert"); a.textContent = msg; a.hidden = !msg; }

/* ---------- settings + sidebar ---------- */
function applySettings() {
  document.documentElement.dataset.theme = settings.theme;
  $("#model").value = settings.model;
  $("#saveHist").checked = settings.save;
  $$('input[name="theme"]').forEach(r => r.checked = r.value === settings.theme);
}
function saveSettings() { store.set("sc.settings", settings); }
$("#model").addEventListener("change", e => { settings.model = e.target.value; saveSettings(); });
$("#saveHist").addEventListener("change", e => { settings.save = e.target.checked; saveSettings(); });
$$('input[name="theme"]').forEach(r => r.addEventListener("change", () => { settings.theme = r.value; saveSettings(); applySettings(); drawWave(); }));

const side = $("#sidebar"), scrim = $("#scrim"), menu = $("#menu");
function toggleSide(open) {
  side.classList.toggle("open", open);
  scrim.hidden = !open;
  menu.setAttribute("aria-expanded", open);
  if (open) $("#model").focus(); else menu.focus();
}
menu.addEventListener("click", () => toggleSide(!side.classList.contains("open")));
scrim.addEventListener("click", () => toggleSide(false));
$("#sideClose").addEventListener("click", () => toggleSide(false));
document.addEventListener("keydown", e => { if (e.key === "Escape" && side.classList.contains("open")) toggleSide(false); });

/* ---------- tabs ---------- */
const tabs = $$('[role="tab"]');
function showTab(name) {
  tabs.forEach(t => { const on = t.dataset.tab === name; t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1; $("#panel-" + t.dataset.tab).hidden = !on; });
  if (name === "progress") renderHistory();
  if (name === "record") sizeWave();
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => showTab(t.dataset.tab));
  t.addEventListener("keydown", e => {
    const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!d) return;
    const n = tabs[(i + d + tabs.length) % tabs.length];
    n.focus(); showTab(n.dataset.tab);
  });
});
$("#goRecord").addEventListener("click", () => showTab("record"));

/* ---------- topics ---------- */
let ti = Math.floor(Math.random() * TOPICS.length);
const setTopic = () => $("#topic").textContent = TOPICS[ti];
$("#newTopic").addEventListener("click", () => { ti = (ti + 1) % TOPICS.length; setTopic(); });

/* ---------- waveform ---------- */
const cv = $("#wave"), cx = cv.getContext("2d");
const levels = [];
let analyser = null, raf = 0;
function sizeWave() {
  const r = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * r; cv.height = 110 * r;
  drawWave();
}
function drawWave() {
  const w = cv.width, hh = cv.height, r = window.devicePixelRatio || 1;
  const cs = getComputedStyle(document.documentElement);
  const bar = 4 * r, gap = 3 * r, n = Math.max(1, Math.floor(w / (bar + gap)));
  cx.clearRect(0, 0, w, hh);
  cx.fillStyle = analyser ? cs.getPropertyValue("--brand") : cs.getPropertyValue("--line");
  for (let i = 0; i < n; i++) {
    const v = levels[levels.length - n + i] || 0;
    const bh = Math.max(4 * r, v * hh * 1.4);
    cx.beginPath();
    cx.roundRect ? cx.roundRect(i * (bar + gap), (hh - bh) / 2, bar, bh, 2 * r) : cx.rect(i * (bar + gap), (hh - bh) / 2, bar, bh);
    cx.fill();
  }
}
function loop() {
  if (!analyser) return;
  const buf = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(buf);
  let sum = 0;
  for (const b of buf) { const x = (b - 128) / 128; sum += x * x; }
  levels.push(Math.min(1, Math.sqrt(sum / buf.length) * 3));
  if (levels.length > 600) levels.splice(0, 300);
  drawWave();
  raf = requestAnimationFrame(loop);
}
window.addEventListener("resize", sizeWave);

/* ---------- recording ---------- */
let rec = null, stream = null, chunks = [], t0 = 0, tick = 0, actx = null;
const recBtn = $("#recBtn");
async function startRec() {
  notify("");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    return notify("This browser can't record audio. Use the Upload tab instead.");
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true } });
  } catch {
    return notify("Microphone access is blocked. Allow it in your browser's address bar, then try again.");
  }
  clearClip();
  chunks = []; levels.length = 0;
  rec = new MediaRecorder(stream);
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  rec.onstop = onRecStop;
  actx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = actx.createAnalyser(); analyser.fftSize = 512;
  actx.createMediaStreamSource(stream).connect(analyser);
  rec.start();
  t0 = performance.now();
  recBtn.classList.add("on"); recBtn.setAttribute("aria-label", "Stop recording");
  $("#recHint").textContent = "Recording. Tap the button when you're done.";
  tick = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    $("#timer").firstChild.textContent = fmtTime(s) + " ";
    if (s >= MAX_REC) stopRec();
  }, 250);
  loop();
}
function stopRec() {
  if (!rec || rec.state === "inactive") return;
  rec.stop();
  stream.getTracks().forEach(t => t.stop());
  clearInterval(tick); cancelAnimationFrame(raf);
  analyser = null; actx && actx.close();
  recBtn.classList.remove("on"); recBtn.setAttribute("aria-label", "Start recording");
  $("#recHint").textContent = "Tap the red button and start talking. It stops by itself at 1 minute.";
  drawWave();
}
async function onRecStop() {
  const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
  await setClip(blob, "Your recording", "recording");
}
recBtn.addEventListener("click", () => (rec && rec.state === "recording") ? stopRec() : startRec());

/* ---------- audio -> 16 kHz mono WAV (no ffmpeg needed) ---------- */
function encodeWav(f32, sr) {
  const b = new ArrayBuffer(44 + f32.length * 2), v = new DataView(b);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); w(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, f32.length * 2, true);
  for (let i = 0; i < f32.length; i++) { const x = Math.max(-1, Math.min(1, f32[i])); v.setInt16(44 + i * 2, x < 0 ? x * 32768 : x * 32767, true); }
  return new Blob([b], { type: "audio/wav" });
}
async function decodeToWav(blob) {
  const data = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const dec = await ctx.decodeAudioData(data);
    const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(dec.duration * 16000)), 16000);
    const src = off.createBufferSource();
    src.buffer = dec; src.connect(off.destination); src.start();
    const out = await off.startRendering();
    return { wav: encodeWav(out.getChannelData(0), 16000), seconds: dec.duration };
  } finally { ctx.close(); }
}

/* ---------- clip handling ---------- */
function clearClip() {
  if (clip) URL.revokeObjectURL(clip.url);
  clip = null;
  $("#ready").hidden = true; $("#results").hidden = true;
  $("#timer").firstChild.textContent = "0:00 ";
}
async function setClip(blob, name, kind) {
  try {
    const { wav, seconds } = await decodeToWav(blob);
    if (seconds < MIN_SEC) return notify(`That clip is only ${seconds.toFixed(1)} seconds. Speak for at least ${MIN_SEC} seconds, ideally 30 to 60.`);
    if (seconds > MAX_FILE_SEC) return notify("That clip is longer than 5 minutes. Trim it and try again.");
    clip = { blob: wav, url: URL.createObjectURL(wav), seconds, name };
    $("#clipName").textContent = name;
    $("#clipMeta").textContent = fmtTime(seconds) + " long" + (seconds < 20 ? " · short clips give less reliable results" : "");
    $("#player").src = clip.url;
    $("#ready").hidden = false;
    $("#timer").firstChild.textContent = fmtTime(seconds) + " ";
    $("#ready").scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (kind === "recording") $("#analyze").focus({ preventScroll: true });
  } catch {
    notify("We couldn't read that audio. Try a WAV or MP3 file, or record again.");
  }
}
const fileIn = $("#file"), drop = $("#drop");
async function takeFile(f) {
  if (!f) return;
  notify("");
  if (!/^audio\//.test(f.type) && !/\.(wav|mp3|m4a|ogg)$/i.test(f.name)) return notify("That isn't an audio file. Use WAV, MP3, M4A or OGG.");
  if (f.size > MAX_MB * 1048576) return notify(`That file is over ${MAX_MB} MB. Use a shorter clip.`);
  clearClip();
  await setClip(f, f.name, "upload");
}
fileIn.addEventListener("change", () => { takeFile(fileIn.files[0]); fileIn.value = ""; });
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => takeFile(e.dataTransfer.files[0]));
$("#discard").addEventListener("click", () => { if (!busy) { clearClip(); levels.length = 0; drawWave(); notify(""); } });
$("#again").addEventListener("click", () => { clearClip(); showTab("record"); window.scrollTo({ top: 0, behavior: "smooth" }); });

/* ---------- analyse ---------- */
function setBusy(on) {
  busy = on;
  $("#analyze").disabled = on; $("#discard").disabled = on;
  $("#analyze .spin").hidden = !on;
  $("#analyzeLbl").textContent = on ? "Analysing…" : "Analyse speech";
  $("#busyNote").hidden = !on;
}
$("#analyze").addEventListener("click", async () => {
  if (!clip || busy) return;
  notify(""); setBusy(true);
  const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), 300000);
  try {
    const fd = new FormData();
    fd.append("audio", clip.blob, "speech.wav");
    fd.append("model", settings.model);
    const r = await fetch("/api/analyze", { method: "POST", body: fd, signal: ctrl.signal });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || "The server had a problem. Please try again.");
    render(data);
    if (settings.save) {
      const hist = store.get("sc.history", []);
      const row = { timestamp: new Date().toISOString() };
      SAVE_KEYS.forEach(k => row[k] = data[k]);
      hist.push(row);
      store.set("sc.history", hist.slice(-200));
    }
  } catch (e) {
    notify(e.name === "AbortError" ? "That took too long. Try a shorter clip or the tiny model." :
      e instanceof TypeError ? "We can't reach the server. Check your connection and try again." : e.message);
  } finally { clearTimeout(timer); setBusy(false); }
});

/* ---------- charts ---------- */
function lineChart(host, vals, { band, aria, xLabel = "" } = {}) {
  host.replaceChildren();
  const W = 420, H = 190, p = { l: 38, r: 12, t: 12, b: 26 };
  let lo = Math.min(...vals, band ? band[0] : Infinity), hi = Math.max(...vals, band ? band[1] : -Infinity);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * .12; lo = Math.max(0, lo - pad); hi += pad;
  const x = i => p.l + (vals.length === 1 ? (W - p.l - p.r) / 2 : i * (W - p.l - p.r) / (vals.length - 1));
  const y = v => p.t + (1 - (v - lo) / (hi - lo)) * (H - p.t - p.b);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": aria, width: "100%" });
  for (let k = 0; k < 3; k++) {
    const v = lo + (hi - lo) * k / 2;
    s.append(svg("line", { class: "grid", x1: p.l, x2: W - p.r, y1: y(v), y2: y(v) }));
    const t = svg("text", { class: "tick", x: p.l - 6, y: y(v) + 3, "text-anchor": "end" }); t.textContent = Math.round(v); s.append(t);
  }
  if (band) s.append(svg("rect", { class: "band", x: p.l, width: W - p.l - p.r, y: y(band[1]), height: y(band[0]) - y(band[1]) }));
  if (vals.length > 1) s.append(svg("polyline", { class: "line", points: vals.map((v, i) => `${x(i)},${y(v)}`).join(" ") }));
  vals.forEach((v, i) => s.append(svg("circle", { class: "dot", cx: x(i), cy: y(v), r: vals.length > 30 ? 2 : 4 })));
  const xl = svg("text", { class: "tick", x: (W + p.l) / 2, y: H - 6, "text-anchor": "middle" });
  xl.textContent = xLabel; s.append(xl);
  host.append(s);
}

/* ---------- results ---------- */
function verdict(sc) {
  return sc >= 85 ? ["Excellent delivery", "Clear, steady and confident. Keep practising to stay sharp."]
    : sc >= 70 ? ["Good job", "Your speech is solid. A couple of small fixes will make it stand out."]
    : sc >= 50 ? ["Getting there", "You have the basics. Focus on the first item under \"What to work on\"."]
    : ["Needs practice", "Don't worry. Pick one habit from the list below and fix just that next time."];
}
function render(d) {
  $("#results").hidden = false;
  const [v, sub] = verdict(d.score);
  $("#verdict").textContent = v; $("#verdictSub").textContent = sub;
  $("#scoreNum").textContent = d.score;
  const fg = $("#ringFg");
  fg.style.strokeDashoffset = 326.7;
  requestAnimationFrame(() => requestAnimationFrame(() => fg.style.strokeDashoffset = 326.7 * (1 - d.score / 100)));

  const stats = [["Words", d.total_words], ["Pace", d.wpm + " WPM"], ["Fillers", d.filler_count], ["Pauses", d.pause_count], ["Length", fmtTime(d.duration_sec)]];
  $("#stats").replaceChildren(...stats.map(([k, val]) => h("div", {}, h("dt", { text: k }), h("dd", { text: String(val) }))));

  const li = (arr, empty) => (arr.length ? arr : [empty]).map(t => h("li", { text: t }));
  $("#good").replaceChildren(...li(d.strengths, "Keep practising. Consistency builds fluency."));
  $("#bad").replaceChildren(...li(d.improvements, "Nothing major to fix. Great job!"));

  const fill = new Set(d._filler_idx), rep = new Set(d._repeat_idx), pau = new Set(d._pause_idx);
  const tr = $("#transcript"); tr.replaceChildren();
  d.words.forEach((w, i) => {
    tr.append(fill.has(i) ? h("mark", { class: "f", text: w }) : rep.has(i) ? h("mark", { class: "r", text: w }) : document.createTextNode(w), " ");
    if (pau.has(i)) tr.append(h("span", { class: "pause", text: "pause" }), " ");
  });

  const fb = Object.entries(d.filler_breakdown).sort((a, b) => b[1] - a[1]);
  const fc = $("#fillerChart"); fc.replaceChildren();
  if (!fb.length) fc.append(h("p", { class: "hint", text: "No filler words detected. Nicely done." }));
  else {
    const max = fb[0][1], box = h("div", { class: "bars" });
    fb.forEach(([k, n]) => { const i = h("i"); box.append(h("div", { class: "bar" }, h("span", { text: k }), h("span", {}, i), h("b", { text: n }))); requestAnimationFrame(() => i.style.width = (n / max * 100) + "%"); });
    fc.append(box);
  }

  const pc = $("#paceChart");
  if (d.pace_per_segment.length) lineChart(pc, d.pace_per_segment, { band: [120, 160], aria: "Words per minute for each segment of speech", xLabel: "Segment" });
  else pc.replaceChildren(h("p", { class: "hint", text: "Not enough speech to show pace over time." }));

  const kv = o => h("div", { class: "kv" }, ...Object.entries(o).map(([k, n]) => h("span", { text: `${k}: ${n}` })));
  const L = $("#lang"); L.replaceChildren(kv({ "Word variety": d.unique_ratio, "Avg. word length": d.avg_word_len, "Avg. sentence length": d.avg_sentence_len, "Repeated words": d.repeat_count }));
  if (Object.keys(d.pos).length) L.append(h("p", { text: "Word types" }), kv(d.pos));
  if (d.overused_words.length) L.append(h("p", { text: "Overused words" }), kv(Object.fromEntries(d.overused_words)));
  if (d.frequent_phrases.length) L.append(h("p", { text: "Repeated phrases" }), kv(Object.fromEntries(d.frequent_phrases)));

  const G = $("#grammar"); G.replaceChildren();
  if (d.grammar_issues === null) G.append(h("p", { text: "The grammar checker isn't available on this server." }));
  else if (!d.grammar_issues.length) G.append(h("p", { text: "No grammar issues found." }));
  else d.grammar_issues.forEach(g => G.append(h("p", { class: "gi" }, h("strong", { text: g.message }), " ", h("em", { text: g.context }))));

  const r = $("#results"); r.scrollIntoView({ behavior: "smooth", block: "start" }); r.focus({ preventScroll: true });
}

/* ---------- history ---------- */
function renderHistory() {
  const hist = store.get("sc.history", []);
  $("#histEmpty").hidden = hist.length > 0; $("#histBody").hidden = !hist.length;
  if (!hist.length) return;
  const x = "Session";
  lineChart($("#hScore"), hist.map(r => r.score), { aria: "Score per session", xLabel: x });
  lineChart($("#hWpm"), hist.map(r => r.wpm), { band: [120, 160], aria: "Words per minute per session", xLabel: x });
  lineChart($("#hFill"), hist.map(r => r.filler_per_100), { aria: "Filler words per 100 words per session", xLabel: x });
  $("#histRows").replaceChildren(...[...hist].reverse().map(r => h("tr", {},
    h("td", { text: new Date(r.timestamp).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) }),
    h("td", { text: r.score }), h("td", { text: r.wpm }), h("td", { text: r.filler_count }), h("td", { text: r.pause_count }), h("td", { text: fmtTime(r.duration_sec) }))));
}
$("#clearHist").addEventListener("click", () => {
  if (confirm("Delete all saved sessions from this device? This can't be undone.")) { store.del("sc.history"); renderHistory(); }
});

/* ---------- footer / dialog ---------- */
$("#yr").textContent = new Date().getFullYear();
const dlg = $("#privacy");
$("#openPrivacy").addEventListener("click", () => dlg.showModal());
$("#closePrivacy").addEventListener("click", () => dlg.close());
dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });
const abt = $("#about");
$$(".open-about").forEach(b => b.addEventListener("click", () => abt.showModal()));
$("#closeAbout").addEventListener("click", () => abt.close());
abt.addEventListener("click", e => { if (e.target === abt) abt.close(); });

applySettings(); setTopic(); sizeWave();
})();