/**
 * Aura — Asistente de inteligencia climática del IDEAM
 * Frontend logic v3
 */

// URLs relativas → funciona en raíz (/) y bajo subpath (/rag/)
const API_BASE = '.';
const SESSION_KEY = 'aura_chat_session';
const HISTORY_KEY = 'aura_chat_history';
const THEME_KEY   = 'aura_chat_theme';

let sessionId = localStorage.getItem(SESSION_KEY) || null;
let isWaiting = false;
let lastUserMessage = '';

// ── DOM ──
const chatArea       = document.getElementById('chatArea');
const welcomeEl      = document.getElementById('welcome');
const messagesEl     = document.getElementById('messages');
const inputEl        = document.getElementById('chatInput');
const sendBtn        = document.getElementById('sendBtn');
const typingEl       = document.getElementById('typingIndicator');
const docCountEl     = document.getElementById('docCountSidebar');
const newChatBtn     = document.getElementById('newChatBtn');
const errorToast     = document.getElementById('errorToast');
const sidebarToggle  = document.getElementById('sidebarToggle');
const sidebar        = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
// Toda celda con data-q es clickeable: quick-cards del welcome + question-items del sidebar
const quickCards     = document.querySelectorAll('[data-q]');

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadHistory();
  inputEl.focus();
});

// ── Events ──
sendBtn.addEventListener('click', sendMessage);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
});

newChatBtn.addEventListener('click', startNewChat);

if (sidebarToggle) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('visible');
  });
}
if (sidebarOverlay) {
  sidebarOverlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');
  });
}

quickCards.forEach(btn => {
  btn.addEventListener('click', () => {
    const q = btn.getAttribute('data-q') || btn.textContent.trim();
    inputEl.value = q;
    sendMessage();
  });
});

// ── Core ──
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isWaiting) return;

  lastUserMessage = text;
  hideWelcome();
  addMessage('user', text);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  setWaiting(true);

  // Cierra sidebar en móvil
  sidebar.classList.remove('open');
  if (sidebarOverlay) sidebarOverlay.classList.remove('visible');

  try {
    const body = { message: text };
    if (sessionId) body.session_id = sessionId;

    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);

    addMessage('agent', data.response);
  } catch (err) {
    addMessage('agent', `⚠️ Error: ${err.message}. Intenta de nuevo.`);
    showError(err.message);
  } finally {
    setWaiting(false);
    inputEl.focus();
  }
}

function hideWelcome() {
  chatArea.classList.remove('chat-empty');
}

function showWelcome() {
  chatArea.classList.add('chat-empty');
}

function addMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const avatar = document.createElement('div');
  if (role === 'user') {
    avatar.className = 'user-avatar';
    avatar.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
  } else {
    avatar.className = 'agent-avatar';
    avatar.innerHTML = '<img src="static/aura-chat.png" alt="Aura">';
  }

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.innerHTML = role === 'agent' ? renderMarkdown(text) : escapeHtml(text);

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);

  scrollToBottom();
  saveHistory();
}

function setWaiting(w) {
  isWaiting = w;
  sendBtn.disabled = w;
  typingEl.classList.toggle('visible', w);
  if (w) scrollToBottom();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// ── Markdown ligero ──
function renderMarkdown(text) {
  let h = escapeHtml(text);
  h = h.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/^[\s]*[-•]\s+(.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
  h = h.replace(/^### (.+)$/gm, '<strong>$1</strong>');
  h = h.replace(/^## (.+)$/gm, '<strong>$1</strong>');
  h = h.replace(/^---$/gm, '<hr>');
  h = h.replace(/\n\n/g, '</p><p>');
  h = h.replace(/\n/g, '<br>');
  return `<p>${h}</p>`;
}

function escapeHtml(t) {
  const d = document.createElement('div');
  d.textContent = t;
  return d.innerHTML;
}

// ── Stats ──
async function loadStats() {
  if (!docCountEl) return;
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (res.ok) {
      const data = await res.json();
      // Mostrar archivos reales (218), no fragmentos (14.444).
      // Fallback a total_documents para compat con backend viejo.
      const count = data.total_files ?? data.total_documents;
      docCountEl.textContent = `${count.toLocaleString('es-CO')} documentos indexados`;
    }
  } catch {
    docCountEl.textContent = 'Sin conexión';
  }
}

// ── Sesión / historia ──
function startNewChat() {
  sessionId = null;
  localStorage.removeItem(SESSION_KEY);
  localStorage.removeItem(HISTORY_KEY);
  messagesEl.innerHTML = '';
  showWelcome();
  sidebar.classList.remove('open');
  if (sidebarOverlay) sidebarOverlay.classList.remove('visible');
  inputEl.focus();
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, messagesEl.innerHTML);
}

function loadHistory() {
  const saved = localStorage.getItem(HISTORY_KEY);
  if (saved && saved.trim()) {
    messagesEl.innerHTML = saved;
    hideWelcome();
    scrollToBottom();
  } else {
    showWelcome();
  }
}

// ── Toast ──
function showError(msg) {
  errorToast.textContent = msg;
  errorToast.classList.add('visible');
  setTimeout(() => errorToast.classList.remove('visible'), 4000);
}
