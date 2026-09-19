// ============== 状态 ==============
let sessions = [];
let currentSid = null;
let streaming = false;

const TOOL_LABELS = {
  search_knowledge: "🔍 检索知识库",
  memory_search: "🧠 回忆长期记忆",
  AnySearch: "🌐 联网搜索",
};

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");

// ============== API ==============
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

// ============== 会话 ==============
async function loadSessions(selectSid = null) {
  const data = await api("/api/sessions");
  sessions = data.sessions;
  renderSessionList();
  if (selectSid) {
    switchSession(selectSid);
  }
}

function renderSessionList() {
  const el = $("session-list");
  el.innerHTML = "";
  sessions.forEach((s) => {
    const div = document.createElement("div");
    div.className = "session-item" + (s.sid === currentSid ? " active" : "");
    const title = document.createElement("span");
    title.textContent = s.title;
    const del = document.createElement("span");
    del.className = "del";
    del.textContent = "✕";
    del.onclick = async (e) => {
      e.stopPropagation();
      await api(`/api/sessions/${s.sid}`, { method: "DELETE" });
      if (s.sid === currentSid) currentSid = null;
      const rest = sessions.filter((x) => x.sid !== s.sid);
      if (rest.length) await loadSessions(rest[0].sid);
      else await newSession();
    };
    div.appendChild(title);
    div.appendChild(del);
    div.onclick = () => switchSession(s.sid);
    el.appendChild(div);
  });
}

async function newSession() {
  const s = await api("/api/sessions", { method: "POST" });
  await loadSessions(s.sid);
}

async function switchSession(sid) {
  if (streaming) return;
  currentSid = sid;
  renderSessionList();
  const data = await api(`/api/sessions/${sid}/messages`);
  $("chat-title").textContent = data.title || "新对话";
  renderHistory(data.messages);
}

function renderHistory(messages) {
  messagesEl.innerHTML = "";
  if (!messages.length) { showEmpty(); return; }
  messages.forEach((m) => appendBubble(m.role, m.content, false));
  scrollBottom();
}

function showEmpty() {
  messagesEl.innerHTML =
    '<div class="empty-tip"><div class="empty-title">开始提问</div>' +
    '<div class="empty-desc">上传文档后可基于资料问答；也能联网搜索、跨会话记住你的偏好。</div>' +
    '<div class="empty-examples">' +
    '<button class="example" data-q="根据我上传的文档，总结一下主要内容">📄 总结上传的文档</button>' +
    '<button class="example" data-q="帮我搜索一下 AgentScope 2.0 的最新动态">🔎 联网搜索最新动态</button>' +
    '<button class="example" data-q="记住我偏好简洁的中文回答">🧠 记住我的回答偏好</button>' +
    "</div></div>";
  messagesEl.querySelectorAll(".example").forEach((b) => {
    b.onclick = () => { inputEl.value = b.dataset.q; autoGrow(); send(); };
  });
}

// ============== 消息渲染 ==============
function appendBubble(role, text, streamingNow) {
  document.querySelector(".empty-tip")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant" && streamingNow) {
    bubble.classList.add("thin");
    bubble.textContent = text;
  } else if (role === "assistant") {
    bubble.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  } else {
    bubble.textContent = text;
  }
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollBottom();
  return bubble;
}

function addToolChip(name, anchor = null) {
  const bar = document.createElement("div");
  bar.className = "tool-chip";
  const span = document.createElement("span");
  span.textContent = TOOL_LABELS[name] || name;
  bar.appendChild(span);
  // 插到当前助手气泡之前，体现「先调用工具、再回答」
  if (anchor) messagesEl.insertBefore(bar, anchor.closest(".msg") || anchor);
  else messagesEl.appendChild(bar);
  scrollBottom();
}

function addNotice(text) {
  const bar = document.createElement("div");
  bar.className = "tool-chip";
  const span = document.createElement("span");
  span.textContent = text;
  bar.appendChild(span);
  messagesEl.appendChild(bar);
  scrollBottom();
}

function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

// ============== 发送 + SSE 流式 ==============
async function send() {
  const text = inputEl.value.trim();
  if (!text || streaming || !currentSid) return;
  streaming = true;
  $("btn-send").disabled = true;
  inputEl.value = "";
  autoGrow();

  appendBubble("user", text, false);
  const bubble = appendBubble("assistant", "", true);
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);

  let full = "";
  const toolSeen = new Set();
  $("tool-status").innerHTML = "";

  try {
    const res = await fetch(`/api/sessions/${currentSid}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) handleFrame(frame);
    }

    function handleFrame(frame) {
      const lines = frame.split("\n");
      let event = "message", data = "";
      lines.forEach((ln) => {
        if (ln.startsWith("event:")) event = ln.slice(6).trim();
        else if (ln.startsWith("data:")) data += ln.slice(5).trim();
      });
      if (!data) return;
      const payload = JSON.parse(data);

      if (event === "tool") {
        if (!toolSeen.has(payload.name)) {
          toolSeen.add(payload.name);
          addToolChip(payload.name, bubble);
          const tag = document.createElement("span");
          tag.textContent = TOOL_LABELS[payload.name] || payload.name;
          $("tool-status").appendChild(tag);
        }
      } else if (event === "token") {
        full += payload.delta;
        bubble.textContent = full;
        bubble.appendChild(cursor);
        scrollBottom();
      } else if (event === "error") {
        full += `\n⚠️ ${payload.message}`;
      } else if (event === "done") {
        if (payload.title) $("chat-title").textContent = payload.title;
      }
    }

    // 结束：渲染 Markdown
    bubble.classList.remove("thin");
    bubble.innerHTML = DOMPurify.sanitize(marked.parse(full || "（无回复）"));
    await loadSessions(currentSid);
  } catch (err) {
    bubble.classList.remove("thin");
    bubble.textContent = "请求失败：" + err.message;
  } finally {
    cursor?.remove();
    streaming = false;
    $("btn-send").disabled = false;
    inputEl.focus();
  }
}

// ============== 上传 / 知识库 ==============
async function refreshDocCount() {
  const data = await api("/api/documents");
  $("doc-count").textContent = data.documents.length;
  const panel = $("doc-panel");
  if (!data.documents.length) {
    panel.innerHTML = '<div class="doc-empty">还没有上传文档</div>';
  } else {
    panel.innerHTML = data.documents
      .map((d) => `<div class="doc-line">📄 ${d.filename} <span style="color:#9aa3b2">· ${d.chunks} 块</span></div>`)
      .join("");
  }
  return data.documents;
}

$("btn-upload").onclick = () => $("file-input").click();
$("file-input").onchange = async (e) => {
  const files = [...e.target.files];
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  $("btn-upload").disabled = true;
  try {
    const data = await api("/api/documents/upload", { method: "POST", body: fd });
    const names = data.documents.map((d) => d.filename).join("、");
    addNotice(`📄 已入库 ${data.documents.length} 个文档：${names}，现在可以基于它们提问`);
    await refreshDocCount();
  } catch (err) {
    addNotice("⚠️ 上传失败：" + err.message);
  } finally {
    $("btn-upload").disabled = false;
    e.target.value = "";
  }
};

$("btn-docs").onclick = async () => {
  const panel = $("doc-panel");
  panel.classList.toggle("hidden");
  if (!panel.classList.contains("hidden")) await refreshDocCount();
};

// ============== 输入交互 ==============
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}
inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
$("btn-send").onclick = send;
$("btn-new").onclick = newSession;

// ============== 启动 ==============
(async function init() {
  await refreshDocCount();
  const data = await api("/api/sessions");
  if (data.sessions.length) await loadSessions(data.sessions[0].sid);
  else await newSession();
})();
