/**
 * RAG Agent Chat — Frontend Logic
 */

const API_BASE = window.location.origin;
const SESSION_KEY = 'rag_chat_session';
const HISTORY_KEY = 'rag_chat_history';

let sessionId = localStorage.getItem(SESSION_KEY) || null;
let isWaiting = false;

// ── DOM Elements ──
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const inputEl = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const typingEl = document.getElementById('typingIndicator');
const docCountEl = document.getElementById('docCount');
const newChatBtn = document.getElementById('newChatBtn');
const errorToast = document.getElementById('errorToast');

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadHistory();
    inputEl.focus();
});

// ── Event Listeners ──
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

document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
        inputEl.value = btn.textContent;
        sendMessage();
    });
});

// ── Core Functions ──

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isWaiting) return;

    // Hide welcome, show message
    if (welcomeEl) welcomeEl.style.display = 'none';

    addMessage('user', text);
    inputEl.value = '';
    inputEl.style.height = 'auto';
    setWaiting(true);

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

function addMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = role === 'agent' ? renderMarkdown(text) : escapeHtml(text);

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' });

    const bubble = document.createElement('div');
    bubble.appendChild(content);
    bubble.appendChild(time);

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);

    scrollToBottom();
    saveHistory();
}

function setWaiting(waiting) {
    isWaiting = waiting;
    sendBtn.disabled = waiting;
    typingEl.classList.toggle('visible', waiting);
    if (waiting) scrollToBottom();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

// ── Markdown Renderer (lightweight) ──

function renderMarkdown(text) {
    let html = escapeHtml(text);

    // Code blocks ```
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold **text**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic *text*
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Links [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    // Unordered lists
    html = html.replace(/^[\s]*[-•]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // Ordered lists
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
    // Headings
    html = html.replace(/^### (.+)$/gm, '<strong>$1</strong>');
    html = html.replace(/^## (.+)$/gm, '<strong>$1</strong>');
    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr style="border-color: var(--border-glass); margin: 8px 0;">');
    // Paragraphs (double newline)
    html = html.replace(/\n\n/g, '</p><p>');
    // Single newlines
    html = html.replace(/\n/g, '<br>');

    return `<p>${html}</p>`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Stats ──

async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (res.ok) {
            const data = await res.json();
            docCountEl.textContent = data.total_documents.toLocaleString();
        }
    } catch {
        docCountEl.textContent = '—';
    }
}

// ── Session Management ──

function startNewChat() {
    sessionId = null;
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(HISTORY_KEY);
    messagesEl.innerHTML = '';
    if (welcomeEl) welcomeEl.style.display = '';
    inputEl.focus();
}

function saveHistory() {
    const messages = messagesEl.innerHTML;
    localStorage.setItem(HISTORY_KEY, messages);
}

function loadHistory() {
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved && saved.trim()) {
        messagesEl.innerHTML = saved;
        if (welcomeEl) welcomeEl.style.display = 'none';
        scrollToBottom();
    }
}

// ── Error Toast ──

function showError(msg) {
    errorToast.textContent = msg;
    errorToast.classList.add('visible');
    setTimeout(() => errorToast.classList.remove('visible'), 4000);
}
