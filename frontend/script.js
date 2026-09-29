// ---------- Config ----------
const API_BASE = window.location.hostname && window.location.hostname !== ""
  ? `${window.location.protocol}//${window.location.hostname}:5000`
  : "http://localhost:5000";

const STORAGE_KEY = "truthlens.history.v1";

// ---------- Animated background ----------
(function animateBackground() {
  const canvas = document.getElementById("bg");
  const ctx = canvas.getContext("2d");
  let w, h, particles;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  function makeParticles() {
    const count = Math.max(28, Math.min(70, Math.floor((w * h) / 24000)));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      r: Math.random() * 1.6 + 0.6,
    }));
  }

  function step() {
    ctx.clearRect(0, 0, w, h);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }

    const linkDist = 130;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < linkDist) {
          ctx.strokeStyle = `rgba(124, 92, 255, ${0.12 * (1 - dist / linkDist)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(198, 188, 255, 0.55)";
      ctx.fill();
    }

    requestAnimationFrame(step);
  }

  window.addEventListener("resize", () => { resize(); makeParticles(); });
  resize();
  makeParticles();
  step();
})();

// ---------- Chat state ----------
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const composerEl = document.getElementById("composer");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const scrollHint = document.getElementById("scrollHint");

let history = loadHistory();

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  } catch {
    /* storage unavailable (private mode, quota) — chat still works, just won't persist */
  }
}

function renderEmptyState() {
  const div = document.createElement("div");
  div.className = "empty-state";
  div.innerHTML = `
    <div class="logo">◈</div>
    <h2>Paste something to fact-check</h2>
    <p>Drop in an article or a claim you're unsure about, and Truthlens will tell you whether it reads as real reporting or fabricated content — with a confidence score.</p>
  `;
  chatEl.appendChild(div);
}

function isNearBottom() {
  return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 80;
}

function scrollToBottom(force = false) {
  if (force || isNearBottom()) {
    chatEl.scrollTop = chatEl.scrollHeight;
    scrollHint.classList.remove("show");
  }
}

chatEl.addEventListener("scroll", () => {
  if (!isNearBottom()) scrollHint.classList.add("show");
  else scrollHint.classList.remove("show");
});

scrollHint.addEventListener("click", () => scrollToBottom(true));

function addUserBubble(text, { persist = true } = {}) {
  const node = document.getElementById("tpl-user").content.cloneNode(true);
  node.querySelector(".bubble").textContent = text;
  chatEl.appendChild(node);
  if (persist) {
    history.push({ role: "user", text });
    saveHistory();
  }
  scrollToBottom(true);
}

function addBotResult(prediction, confidence, { persist = true } = {}) {
  const node = document.getElementById("tpl-bot").content.cloneNode(true);
  const isReal = prediction === "REAL";

  node.querySelector(".verdict-label").textContent = "Verdict:";
  const tag = node.querySelector(".verdict-tag");
  tag.textContent = prediction;
  tag.classList.add(isReal ? "real" : "fake");

  const realPct = Math.round((confidence?.REAL ?? 0) * 100);
  const fakePct = Math.round((confidence?.FAKE ?? 0) * 100);

  chatEl.appendChild(node);

  const mounted = chatEl.lastElementChild;
  const realFill = mounted.querySelector(".bar-fill--real");
  const fakeFill = mounted.querySelector(".bar-fill--fake");
  mounted.querySelector(".pct-real").textContent = `${realPct}%`;
  mounted.querySelector(".pct-fake").textContent = `${fakePct}%`;
  requestAnimationFrame(() => {
    realFill.style.width = `${realPct}%`;
    fakeFill.style.width = `${fakePct}%`;
  });

  if (persist) {
    history.push({ role: "bot", prediction, confidence });
    saveHistory();
  }
  scrollToBottom();
}

function addErrorBubble(message, { persist = true } = {}) {
  const node = document.getElementById("tpl-error").content.cloneNode(true);
  node.querySelector(".bubble").textContent = message;
  chatEl.appendChild(node);
  if (persist) {
    history.push({ role: "error", text: message });
    saveHistory();
  }
  scrollToBottom();
}

function addTyping() {
  const node = document.getElementById("tpl-typing").content.cloneNode(true);
  chatEl.appendChild(node);
  const el = chatEl.lastElementChild;
  scrollToBottom(true);
  return () => el.remove();
}

function renderHistory() {
  chatEl.innerHTML = "";
  if (history.length === 0) {
    renderEmptyState();
    return;
  }
  for (const item of history) {
    if (item.role === "user") addUserBubble(item.text, { persist: false });
    else if (item.role === "bot") addBotResult(item.prediction, item.confidence, { persist: false });
    else if (item.role === "error") addErrorBubble(item.text, { persist: false });
  }
  scrollToBottom(true);
}

// ---------- API ----------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) throw new Error();
    statusEl.textContent = "API connected";
    statusEl.className = "status status--ok";
  } catch {
    statusEl.textContent = "API unreachable";
    statusEl.className = "status status--down";
  }
}

async function analyze(text) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ news_text: text }),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok || !body || body.error) {
    throw new Error(body?.error || `Request failed (${res.status})`);
  }
  return body;
}

// ---------- Composer behavior ----------
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}
inputEl.addEventListener("input", autoGrow);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerEl.requestSubmit();
  }
});

composerEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  if (history.length === 0) chatEl.innerHTML = ""; // clear empty-state

  addUserBubble(text);
  inputEl.value = "";
  autoGrow();
  sendBtn.disabled = true;

  const stopTyping = addTyping();
  try {
    const result = await analyze(text);
    stopTyping();
    addBotResult(result.prediction, result.confidence);
  } catch (err) {
    stopTyping();
    addErrorBubble(err.message || "Something went wrong while analyzing this.");
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
});

clearBtn.addEventListener("click", () => {
  history = [];
  saveHistory();
  renderHistory();
});

// ---------- Init ----------
renderHistory();
checkHealth();
setInterval(checkHealth, 15000);
