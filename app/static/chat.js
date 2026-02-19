/**
 * RAG Agent Chat — Frontend Logic v2
 */

const API_BASE = window.location.origin;
const SESSION_KEY = 'rag_chat_session';
const HISTORY_KEY = 'rag_chat_history';
const RECENT_KEY = 'rag_chat_recent';
const THEME_KEY = 'rag_chat_theme';

let sessionId = localStorage.getItem(SESSION_KEY) || null;
let isWaiting = false;

// ── DOM ──
const chatArea = document.getElementById('chatArea');
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const typingEl = document.getElementById('typingIndicator');
const docCountEl = document.getElementById('docCountSidebar');
const newChatBtn = document.getElementById('newChatBtn');
const errorToast = document.getElementById('errorToast');
const recentList = document.getElementById('recentList');
const themeToggle = document.getElementById('themeToggle');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const regenerateBtn = document.getElementById('regenerateBtn');

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    loadStats();
    loadHistory();
    loadRecent();
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
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

newChatBtn.addEventListener('click', startNewChat);

themeToggle.addEventListener('click', toggleTheme);

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('visible');
});

sidebarOverlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');
});

document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
        inputEl.value = btn.textContent;
        sendMessage();
    });
});

// ── Core ──
let lastUserMessage = '';

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isWaiting) return;

    lastUserMessage = text;
    addMessage('user', text);
    addToRecent(text);
    inputEl.value = '';
    inputEl.style.height = 'auto';
    setWaiting(true);

    // Close sidebar on mobile
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');

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
        regenerateBtn.style.display = 'inline-block';
    } catch (err) {
        addMessage('agent', `⚠️ Error: ${err.message}. Intenta de nuevo.`);
        showError(err.message);
    } finally {
        setWaiting(false);
        inputEl.focus();
    }
}

function addMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? '👤' : '<img src="/static/avatar.png" alt="Bot" class="avatar-img">';

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

// ── Markdown ──
function renderMarkdown(text) {
    let h = escapeHtml(text);
    h = h.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
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

// ── Recent Queries ──
function addToRecent(text) {
    let recent = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
    recent = recent.filter(r => r !== text);
    recent.unshift(text);
    recent = recent.slice(0, 6);
    localStorage.setItem(RECENT_KEY, JSON.stringify(recent));
    renderRecent(recent);
}

function loadRecent() {
    const recent = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
    renderRecent(recent);
}

function renderRecent(recent) {
    if (!recent.length) {
        recentList.innerHTML = '<li class="recent-empty">Aún no hay consultas</li>';
        return;
    }
    recentList.innerHTML = recent.map(q =>
        `<li title="${escapeHtml(q)}">${escapeHtml(q.length > 35 ? q.slice(0, 35) + '…' : q)}</li>`
    ).join('');

    recentList.querySelectorAll('li').forEach((li, i) => {
        li.addEventListener('click', () => {
            inputEl.value = recent[i];
            sendMessage();
        });
    });
}

// ── Theme ──
function loadTheme() {
    const theme = localStorage.getItem(THEME_KEY) || 'light';
    document.documentElement.setAttribute('data-theme', theme);
    themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(THEME_KEY, next);
    themeToggle.textContent = next === 'dark' ? '☀️' : '🌙';
}

// ── Stats ──
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (res.ok) {
            const data = await res.json();
            docCountEl.textContent = `${data.total_documents.toLocaleString()} documentos indexados`;
        }
    } catch {
        docCountEl.textContent = 'Sin conexión';
    }
}

// ── Session ──
function startNewChat() {
    sessionId = null;
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(HISTORY_KEY);

    // Keep only the welcome message
    messagesEl.innerHTML = `
    <div class="message agent">
      <div class="message-avatar"><img src="/static/avatar.png" alt="Bot" class="avatar-img"></div>
      <div class="message-bubble">
        <p>¡Hola! Soy el asistente inteligente. ¿En qué puedo ayudarte hoy?</p>
      </div>
    </div>`;

    regenerateBtn.style.display = 'none';
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');
    inputEl.focus();
}

function saveHistory() {
    localStorage.setItem(HISTORY_KEY, messagesEl.innerHTML);
}

function loadHistory() {
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved && saved.trim()) {
        messagesEl.innerHTML = saved;
        scrollToBottom();
    }
}

// ── Regenerate ──
if (regenerateBtn) {
    regenerateBtn.addEventListener('click', () => {
        if (lastUserMessage && !isWaiting) {
            inputEl.value = lastUserMessage;
            sendMessage();
        }
    });
}

// ── Error Toast ──
function showError(msg) {
    errorToast.textContent = msg;
    errorToast.classList.add('visible');
    setTimeout(() => errorToast.classList.remove('visible'), 4000);
}
