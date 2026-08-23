/**
 * ai-manager.js — Полная замена AI‑чата на многопользовательскую систему.
 * Поддерживает множество сессий, авто‑названия, историю, переключение.
 * Подключать после всех основных скриптов (ui.js, actions.js, AiChat.js).
 */
(function() {
    if (window._aiManagerLoaded) return;
    window._aiManagerLoaded = true;

    // =================================================================
    // 1. РАБОТА С СЕССИЯМИ
    // =================================================================
    function getAiSessions() {
        try { return JSON.parse(localStorage.getItem('ai_sessions') || '[]'); }
        catch { return []; }
    }
    function saveAiSessions(sessions) {
        localStorage.setItem('ai_sessions', JSON.stringify(sessions));
    }
    function generateAiSessionId() {
        return 'ai_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
    }
    function createAiSession(firstMessage = '') {
        const sessions = getAiSessions();
        const id = generateAiSessionId();
        let name = firstMessage ? firstMessage.slice(0, 30) : 'AI Chat #' + (sessions.length + 1);
        if (name.length < 3) name = 'AI Chat #' + (sessions.length + 1);
        sessions.push({ id, name, created: Date.now() });
        saveAiSessions(sessions);
        return id;
    }
    function updateAiSessionName(sessionId, newName) {
        const sessions = getAiSessions();
        const found = sessions.find(s => s.id === sessionId);
        if (found) {
            found.name = newName.slice(0, 30);
            saveAiSessions(sessions);
            const item = document.querySelector(`.conversation-item[data-address="${sessionId}"]`);
            if (item) {
                const nameEl = item.querySelector('.name');
                if (nameEl) nameEl.textContent = found.name;
            }
        }
    }
    function loadAiSessionsIntoConversations() {
        const sessions = getAiSessions();
        const container = document.getElementById('conversationsList');
        if (!container) return;
        // Удаляем старые AI-элементы
        container.querySelectorAll('.conversation-item[data-address^="ai_"]').forEach(el => el.remove());
        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            item.dataset.address = session.id;
            item.dataset.isGroup = '0';
            const initials = 'AI';
            item.innerHTML = `
                <div class="avatar">${initials}</div>
                <div class="info">
                    <div class="name truncate">${Utils.escapeHtml(session.name)}</div>
                    <div class="meta"><span class="status"></span><span class="truncate">🤖 AI session</span></div>
                </div>
            `;
            item.onclick = ((addr) => () => window.selectConversation(addr, session.name, false))(session.id);
            container.appendChild(item);
        });
    }

    // =================================================================
    // 2. ПЕРЕОПРЕДЕЛЕНИЕ selectConversation
    // =================================================================
    const _origSelectConversation = window.selectConversation;
    window.selectConversation = function(address, name, isGroup) {
        if (address && address.startsWith('ai_')) {
            // Наша логика для AI-сессий
            if (window.State) {
                window.State.currentChatAddress = address;
                window.State.currentChatIsGroup = false;
                window.State.currentChatPartnerAddress = '';
            }
            const mainContainer = document.getElementById('messagesContainer');
            const mainInputArea = document.querySelector('.chat-panel .input-area:not(#aiChatContainer .input-area)');
            const mainChatHeader = document.querySelector('.chat-panel .chat-panel-header:not(#aiChatContainer .chat-panel-header)');
            const aiContainer = document.getElementById('aiChatContainer');

            if (mainContainer) mainContainer.style.display = 'none';
            if (mainInputArea) mainInputArea.style.display = 'none';
            if (mainChatHeader) mainChatHeader.style.display = 'none';
            if (aiContainer) {
                aiContainer.classList.remove('hidden');
                if (typeof window._initAiChatSession === 'function') {
                    window._initAiChatSession(address);
                }
            }
            const displayName = name || (address.slice(3, 8) + '…');
            const nameEl = document.getElementById('currentChatName');
            if (nameEl) nameEl.textContent = '🤖 ' + displayName;
            const subtitleEl = document.getElementById('chatSubtitle');
            if (subtitleEl) subtitleEl.textContent = 'AI Assistant';

            // Включаем элементы управления
            if (typeof window._enableChatControls === 'function') window._enableChatControls();
            document.querySelectorAll('.conversation-item').forEach(item => item.classList.remove('active'));
            const activeItem = document.querySelector(`.conversation-item[data-address="${address}"]`);
            if (activeItem) activeItem.classList.add('active');

            // ========== ФИКС: для мобильных устройств показываем чат-панель ==========
            if (window.innerWidth < 768 && typeof window.showChatPanel === 'function') {
                window.showChatPanel();
            }

            return;
        }
        // Не AI – вызываем оригинал
        if (_origSelectConversation) _origSelectConversation(address, name, isGroup);
    };

    // =================================================================
    // 3. ПОЛНАЯ ЗАМЕНА AI-ЧАТА (на основе AiChat.js с сессиями)
    // =================================================================
    // ─── переменные состояния ───
    let _aiChatActive = false;
    let _pendingImageFile = null;
    let _currentStreamingMessage = null;
    let _currentStreamingText = '';
    let _currentStreamReader = null;
    let _isSending = false;
    let _currentImagePreviewUrl = null;
    let _reasoningEnabled = false;
    let _internetEnabled = false;
    let _aiMessagesContainer = null;
    let _aiMessageInput = null;
    let _aiSendBtn = null;
    let _aiAttachBtn = null;
    let _aiReasoningBtn = null;
    let _aiInternetBtn = null;
    let _aiImageInput = null;
    let _aiClearHistoryBtn = null;
    let _closeAiChatBtn = null;
    let _aiImageGenBtn = null;
    let _aiStopBtn = null;
    let _aiNewChatBtn = null;
    let _currentAbortController = null;
    let _currentAiSessionId = null;
    let _aiNameSet = false;

    const CONFIG = {
        historyMaxLength: 200,
        imageMaxWidth: 800,
        imageQuality: 0.7,
        apiEndpoint: '/ai/chat',
        searchEndpoint: '/ai/search',
    };

    // ─── работа с историей сессии ───
    function _getStoredHistory(sessionId) {
        const key = 'ai_chat_history_' + (sessionId || 'default');
        try { return JSON.parse(localStorage.getItem(key) || '[]'); }
        catch { return []; }
    }
    function _setStoredHistory(history, sessionId) {
        const key = 'ai_chat_history_' + (sessionId || 'default');
        try { localStorage.setItem(key, JSON.stringify(history)); }
        catch {}
    }
    function _saveAiMessage(role, text, sessionId) {
        let history = _getStoredHistory(sessionId);
        history.push({ role, text, timestamp: Date.now() });
        if (history.length > CONFIG.historyMaxLength) history = history.slice(-CONFIG.historyMaxLength);
        _setStoredHistory(history, sessionId);
    }
    function _loadAiHistory(sessionId) {
        if (!_aiMessagesContainer) return;
        const history = _getStoredHistory(sessionId);
        _aiMessagesContainer.innerHTML = '';
        history.forEach(msg => _displayAiMessage(msg.text, msg.role === 'user', null, false));
        if (history.length === 0) _displayWelcome();
    }
    // В ai-manager.js добавить:
function removeAiSession(sessionId) {
    let sessions = getAiSessions();
    sessions = sessions.filter(s => s.id !== sessionId);
    saveAiSessions(sessions);
    // Удаляем элемент из DOM
    const item = document.querySelector(`.conversation-item[data-address="${sessionId}"]`);
    if (item) item.remove();
}

// В _clearAiHistory, после очистки истории, вызываем removeAiSession:
function _clearAiHistory() {
    if (!_currentAiSessionId) {
        _showToast('No active AI session', 'warning');
        return;
    }
    if (typeof window.showConfirmModal === 'function') {
        window.showConfirmModal('Clear AI History', 'This will delete the chat and remove it from the list. Continue?')
            .then((confirmed) => {
                if (confirmed) {
                    const sessionId = _currentAiSessionId;
                    // 1. Удаляем историю сообщений
                    localStorage.removeItem('ai_chat_history_' + sessionId);

                    // 2. Удаляем сессию из списка (из localStorage и из DOM)
                    removeAiSession(sessionId);

                    // 3. Переключаемся на первый обычный чат (если есть)
                    const firstConv = document.querySelector('.conversation-item:not([data-address^="ai_"])');
                    if (firstConv?.dataset.address) {
                        window.selectConversation(firstConv.dataset.address, '', firstConv.dataset.isGroup === '1');
                    } else {
                        // 4. Если обычных чатов нет – переходим в пустое состояние
                        // Сбрасываем текущий чат в State
                        if (window.State) {
                            window.State.currentChatAddress = '';
                            window.State.currentChatIsGroup = false;
                            window.State.currentChatPartnerAddress = '';
                        }

                        // Показываем пустое состояние в основном контейнере сообщений
                        const mainContainer = document.getElementById('messagesContainer');
                        if (mainContainer) {
                            mainContainer.innerHTML = `
                                <div class="empty-state">
                                    <div class="icon">👋</div>
                                    <p data-i18n="select_conversation">Select a conversation to start chatting</p>
                                </div>
                            `;
                            mainContainer.style.display = 'flex';
                        }

                        // Скрываем AI-контейнер, показываем обычные элементы чата
                        const aiContainer = document.getElementById('aiChatContainer');
                        if (aiContainer) aiContainer.classList.add('hidden');

                        const mainChatHeader = document.querySelector('.chat-panel .chat-panel-header:not(#aiChatContainer .chat-panel-header)');
                        const mainInputArea = document.querySelector('.chat-panel .input-area:not(#aiChatContainer .input-area)');
                        if (mainChatHeader) mainChatHeader.style.display = 'flex';
                        if (mainInputArea) mainInputArea.style.display = 'flex';

                        // Обновляем заголовок чата
                        const nameEl = document.getElementById('currentChatName');
                        if (nameEl) nameEl.textContent = 'Select a conversation';
                        const subtitleEl = document.getElementById('chatSubtitle');
                        if (subtitleEl) subtitleEl.textContent = '';

                        // Снимаем активность со всех элементов в списке
                        document.querySelectorAll('.conversation-item').forEach(item => item.classList.remove('active'));

                        // Если мобильное устройство – показываем список бесед
                        if (window.innerWidth < 768 && typeof window.showConversationsList === 'function') {
                            window.showConversationsList();
                        }
                    }
                    _showToast('Chat deleted', 'success');
                }
            });
    }
}

    // ─── вспомогательные ───
    function _escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]))
                  .replace(/['"]/g, m => ({ "'": '&#39;', '"': '&quot;' }[m]));
    }
    function _showToast(message, type = 'info') {
        if (window.NotificationManager?.showToast) window.NotificationManager.showToast(message, type);
        else console.log(`[${type}] ${message}`);
    }
    function _attachReasoningToggle(container) {
        if (!container) return;
        container.querySelectorAll('.reasoning-block details').forEach(details => {
            const summary = details.querySelector('summary');
            if (summary && !summary._listenerAttached) {
                summary._listenerAttached = true;
                const toggle = function(e) {
                    e.preventDefault();
                    const isOpen = details.getAttribute('open') !== null;
                    if (isOpen) details.removeAttribute('open');
                    else details.setAttribute('open', 'open');
                };
                summary.addEventListener('click', toggle);
                summary.addEventListener('touchstart', toggle, { passive: false });
                summary.addEventListener('pointerdown', toggle);
            }
        });
    }
    function _enhanceCodeBlocks(container) {
        if (!container || typeof hljs === 'undefined') return;
        container.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        container.querySelectorAll('pre').forEach(pre => {
            if (pre.querySelector('.copy-code-btn')) return;
            const btn = document.createElement('button');
            btn.textContent = '📋 Copy';
            btn.className = 'copy-code-btn';
            btn.style.cssText = 'position:absolute;top:8px;right:8px;background:#3c3c3c;border:none;color:#ccc;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px;z-index:1;';
            btn.onclick = (e) => {
                e.stopPropagation();
                const code = pre.querySelector('code');
                if (!code) return;
                navigator.clipboard.writeText(code.innerText).then(() => {
                    btn.textContent = '✅ Copied!';
                    setTimeout(() => btn.textContent = '📋 Copy', 2000);
                });
            };
            pre.style.position = 'relative';
            pre.style.paddingTop = '32px';
            pre.appendChild(btn);
        });
    }
    function _addImageDownloadButtons(container) {
        if (!container) return;
        container.querySelectorAll('img').forEach(img => {
            if (img.hasAttribute('data-modal-attached')) return;
            img.setAttribute('data-modal-attached', 'true');
            img.style.cursor = 'pointer';
            img.addEventListener('click', (e) => {
                e.stopPropagation();
                if (typeof window.openImageModal === 'function') {
                    window.openImageModal(img.src);
                }
            });
            if (img.parentNode.querySelector('.download-image-btn')) return;
            const imageUrl = img.src;
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'position:relative; display:inline-block; margin:0 4px 8px 0;';
            const parent = img.parentNode;
            const imgClone = img.cloneNode(true);
            imgClone.style.cursor = 'pointer';
            imgClone.addEventListener('click', (e) => {
                e.stopPropagation();
                window.openImageModal(imgClone.src);
            });
            const downloadBtn = document.createElement('button');
            downloadBtn.textContent = '💾 Скачать';
            downloadBtn.className = 'download-image-btn';
            downloadBtn.style.cssText = 'display:block; margin-top:4px; background:var(--accent-soft); border:none; border-radius:6px; padding:2px 8px; font-size:11px; cursor:pointer; color:var(--accent); width:100%; text-align:center;';
            downloadBtn.onclick = (e) => {
                e.stopPropagation();
                _downloadImage(imageUrl, 'generated_image.png');
            };
            wrapper.appendChild(imgClone);
            wrapper.appendChild(downloadBtn);
            parent.replaceChild(wrapper, img);
        });
    }
    async function _downloadImage(imageUrl, filename = 'image.png') {
        try {
            const response = await fetch(imageUrl);
            if (!response.ok) throw new Error('Network error');
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            _showToast('Изображение сохранено', 'success');
        } catch (err) {
            console.error('Ошибка скачивания:', err);
            _showToast('Не удалось сохранить изображение', 'error');
        }
    }
    function _renderMarkdown(text) {
        if (!text) return '';
        try {
            const reasoningRegex = /💭\s*РАССУЖДЕНИЕ:\s*([\s\S]*?)\s*---/i;
            let mainText = text;
            let reasoningHtml = '';
            const match = reasoningRegex.exec(text);
            if (match) {
                const reasoningContent = match[1].trim();
                reasoningHtml = `
                    <div class="reasoning-block">
                        <details>
                            <summary>💭 Reasoning</summary>
                            <div class="reasoning-content">${marked.parse(reasoningContent)}</div>
                        </details>
                    </div>
                `;
                mainText = text.replace(match[0], '').trim();
            }
            let html = marked.parse(mainText);
            if (reasoningHtml) html = reasoningHtml + html;
            return DOMPurify.sanitize(html);
        } catch (e) {
            return _escapeHtml(text);
        }
    }
    function _displayWelcome() {
        _displayAiMessage(
            'Привет! Я AI-ассистент с **автоматической загрузкой нескольких страниц из интернета**.\n\n' +
            '- 🌐 Кнопка "Интернет" включает поиск и чтение целых страниц (до 3 результатов)\n' +
            '- 🧠 Режим рассуждений показывает ход мыслей\n' +
            '- 🔗 Вставь ссылку — я прочитаю содержимое\n' +
            '- 📎 Прикрепи изображение для анализа\n' +
            '- 🎨 Генерация изображений — кнопка рядом с полем ввода',
            false, null, false
        );
    }
    async function _compressImage(dataUrl, maxWidth = CONFIG.imageMaxWidth, quality = CONFIG.imageQuality) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width, height = img.height;
                if (width > maxWidth || height > maxWidth) {
                    if (width > height) { height = height * (maxWidth / width); width = maxWidth; }
                    else { width = width * (maxWidth / height); height = maxWidth; }
                }
                canvas.width = width; canvas.height = height;
                canvas.getContext('2d').drawImage(img, 0, 0, width, height);
                resolve(canvas.toDataURL('image/jpeg', quality));
            };
            img.onerror = reject;
            img.src = dataUrl;
        });
    }
    function _showAiTypingIndicator(show, statusText = '') {
        if (!_aiMessagesContainer) return;
        let indicator = _aiMessagesContainer.querySelector('.typing-indicator-message');
        if (show) {
            if (indicator) indicator.remove();
            indicator = document.createElement('div');
            indicator.className = 'message received typing-indicator-message';
            if (statusText) {
                indicator.innerHTML = `
                    <div class="avatar">🤖</div>
                    <div class="content">
                        <div style="color:var(--text-secondary);font-size:13px;padding:8px 0;">${_escapeHtml(statusText)}</div>
                    </div>`;
            } else {
                indicator.innerHTML = `
                    <div class="avatar">🤖</div>
                    <div class="typing-indicator"><span></span><span></span><span></span></div>`;
            }
            _aiMessagesContainer.appendChild(indicator);
            _aiMessagesContainer.scrollTop = _aiMessagesContainer.scrollHeight;
        } else {
            if (indicator) indicator.remove();
        }
    }
    function _displaySearchSources(results) {
        if (!results || !results.length) return;
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'search-sources animate-fade';
        sourcesDiv.style.cssText = `
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 6px 0 6px 36px;
            font-size: 12px;
            color: var(--text-secondary);
        `;
        const links = results
            .filter(r => r.url)
            .map(r => `<a href="${_escapeHtml(r.url)}" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;display:block;margin:2px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${_escapeHtml(r.title || r.url)}">${_escapeHtml((r.title || r.url).slice(0, 80))}</a>`)
            .join('');
        sourcesDiv.innerHTML = `<span style="opacity:.7;">🔍 Источники:</span><div style="margin-top:4px;">${links || '<span style="opacity:.6;">нет прямых ссылок</span>'}</div>`;
        if (_aiMessagesContainer) {
            _aiMessagesContainer.appendChild(sourcesDiv);
            _aiMessagesContainer.scrollTop = _aiMessagesContainer.scrollHeight;
        }
    }
    function _displayResearchResult(data) {
        const confidencePercent = (data.confidence * 100).toFixed(0);
        let header = `🔬 **Исследование** | Уверенность: ${confidencePercent}%`;
        let hypoList = Array.isArray(data.hypotheses)
            ? data.hypotheses.map(h => `- ${h}`).join('\n')
            : '*(гипотезы не сгенерированы)*';
        let evidenceText = '';
        if (data.evidence && data.evidence.length) {
            evidenceText = '\n\n**Извлечённые факты:**\n' + data.evidence.map(e => {
                return `- ${e.extracted_facts || (e.web_evidence ? e.web_evidence.slice(0, 150) : '')}`;
            }).join('\n');
        }
        const full = `${header}\n\n**Гипотезы:**\n${hypoList}\n\n**Вывод:**\n${data.answer || 'Нет ответа'}${evidenceText}`;
        _displayAiMessage(full, false, null, true);
    }
    async function _sendResearchQuery(messageText) {
        _showAiTypingIndicator(true, '🔬 Формулирую гипотезы и проверяю факты...');
        try {
            const response = await fetch('/ai/research', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ goal: messageText })
            });
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Research API error: ${response.status} ${errText}`);
            }
            const data = await response.json();
            _showAiTypingIndicator(false);
            _displayResearchResult(data);
        } catch (err) {
            console.error('Research error:', err);
            _showAiTypingIndicator(false);
            _displayAiMessage(`❌ Ошибка исследования: ${err.message}`, false, null, true);
        }
    }
    function _displayAiMessage(text, isUser, imagePreview = null, saveToStorage = true) {
    // <-- ИЗМЕНЕНИЕ: проверка и инициализация контейнера
    if (!_aiMessagesContainer) {
        _aiMessagesContainer = document.getElementById('aiMessagesContainer');
        if (!_aiMessagesContainer) return; // если всё ещё нет – выходим
    }
    // <-- КОНЕЦ ИЗМЕНЕНИЯ

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'sent' : 'received'} animate-fade`;
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    let imageHtml = '';
    if (imagePreview) {
        imageHtml = `<img src="${_escapeHtml(imagePreview)}" alt="Attached" style="max-width:200px;max-height:150px;border-radius:8px;margin-bottom:8px;cursor:pointer;" onclick="window.openImageModal && window.openImageModal('${_escapeHtml(imagePreview)}')">`;
    }
    if (isUser) {
        msgDiv.innerHTML = `
            <div class="avatar">👤</div>
            <div class="content">
                ${imageHtml}
                <div class="markdown-body">${_escapeHtml(text)}</div>
                <div class="meta"><span>${time}</span></div>
            </div>`;
    } else {
        const html = _renderMarkdown(text);
        msgDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="content">
                ${imageHtml}
                <div class="markdown-body">${html}</div>
                <div class="meta"><span>${time}</span></div>
            </div>`;
        const markdownBody = msgDiv.querySelector('.markdown-body');
        if (markdownBody) {
            _enhanceCodeBlocks(markdownBody);
            _addImageDownloadButtons(markdownBody);
            _attachReasoningToggle(markdownBody);
        }
    }
    _aiMessagesContainer.appendChild(msgDiv);
    _aiMessagesContainer.scrollTop = _aiMessagesContainer.scrollHeight;
    if (saveToStorage && text && !text.includes('Привет! Я AI-ассистент')) {
        _saveAiMessage(isUser ? 'user' : 'assistant', text, _currentAiSessionId);
    }
    return msgDiv;
}

    // ─── основная отправка ───
    async function _sendToAi(messageText, imageFile) {
    // Проверка контейнера
    if (!_aiMessagesContainer) {
        _aiMessagesContainer = document.getElementById('aiMessagesContainer');
        if (!_aiMessagesContainer && _currentAiSessionId) {
            _initAiChatSession(_currentAiSessionId);
            _aiMessagesContainer = document.getElementById('aiMessagesContainer');
        }
        if (!_aiMessagesContainer) {
            _showToast('Ошибка: контейнер чата не найден', 'error');
            return;
        }
    }

    if (_isSending) { _showToast('Подождите, предыдущий запрос обрабатывается', 'warning'); return; }
    if (!messageText.trim() && !imageFile) { _showToast('Введите сообщение или выберите изображение', 'warning'); return; }

    // Авто-название
    if (_currentAiSessionId && !_aiNameSet) {
        const firstMsg = messageText.trim();
        if (firstMsg) {
            const newName = firstMsg.slice(0, 30);
            if (newName.length > 2) {
                if (typeof updateAiSessionName === 'function') {
                    updateAiSessionName(_currentAiSessionId, newName);
                }
                _aiNameSet = true;
            }
        }
    }

    // Исследовательский режим
    const researchKeywords = ['правда ли', 'докажи', 'опровергни', 'исследуй', 'проверь', 'действительно ли'];
    const isResearch = researchKeywords.some(kw => messageText.toLowerCase().includes(kw));

    if (isResearch && !imageFile) {
        _aiMessageInput.disabled = true;
        _aiSendBtn.disabled = true;
        _aiAttachBtn.disabled = true;
        if (_aiImageGenBtn) _aiImageGenBtn.disabled = true;
        if (_aiInternetBtn) _aiInternetBtn.disabled = true;
        if (_aiReasoningBtn) _aiReasoningBtn.disabled = true;
        if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = true;
        if (_aiStopBtn) _aiStopBtn.disabled = true;

        const originalText = messageText;
        _aiMessageInput.value = '';
        _clearImagePreview();
        _displayAiMessage(originalText, true, null, true);
        try {
            await _sendResearchQuery(originalText);
        } finally {
            _aiMessageInput.disabled = false;
            _aiSendBtn.disabled = false;
            _aiAttachBtn.disabled = false;
            if (_aiImageGenBtn) _aiImageGenBtn.disabled = false;
            if (_aiInternetBtn) _aiInternetBtn.disabled = false;
            if (_aiReasoningBtn) _aiReasoningBtn.disabled = false;
            if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = false;
            if (_aiStopBtn) _aiStopBtn.disabled = false;
            _aiMessageInput.focus();
            _isSending = false;
        }
        return;
    }

    // ---- ОСНОВНОЙ ПОТОК (обычный чат) ----

    if (_currentStreamReader) {
        try { _currentStreamReader.cancel(); } catch(e) {}
        _currentStreamReader = null;
    }
    _currentStreamingMessage = null;
    _currentStreamingText = '';
    _isSending = true;

    if (_aiSendBtn) _aiSendBtn.style.display = 'none';
    if (_aiStopBtn) _aiStopBtn.style.display = 'inline-flex';

    const urlMatch = messageText.match(/https?:\/\/[^\s]+/);
    const urlToFetch = urlMatch ? urlMatch[0] : null;

    let imageBase64 = null, imageMime = null, compressedDataUrl = null;
    if (imageFile) {
        const reader = new FileReader();
        compressedDataUrl = await new Promise((resolve) => {
            reader.onload = async (e) => resolve(await _compressImage(e.target.result));
            reader.readAsDataURL(imageFile);
        });
        const parts = compressedDataUrl.split(',');
        imageBase64 = parts[1];
        const mimeMatch = parts[0].match(/^data:(image\/[a-zA-Z]+);?/);
        imageMime = mimeMatch ? mimeMatch[1] : 'image/jpeg';
        _displayAiMessage(messageText || '📷 Изображение', true, compressedDataUrl, true);
    } else {
        _displayAiMessage(messageText, true, null, true);
    }

    const useWebSearch = _internetEnabled && !urlToFetch;
    const isSearching = useWebSearch || !!urlToFetch;
    _showAiTypingIndicator(true, isSearching ? (urlToFetch ? `🔗 Загружаю ${urlToFetch.slice(0, 50)}…` : '🔍 Ищу в интернете и загружаю страницы…') : '');

    _currentAbortController = new AbortController();
    const signal = _currentAbortController.signal;

    try {
        const response = await fetch(CONFIG.apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: messageText,
                image_base64: imageBase64,
                image_mime: imageMime,
                stream: true,
                reasoning: _reasoningEnabled,
                web_search: useWebSearch,
                url_to_fetch: urlToFetch,
            }),
            signal: signal
        });
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `HTTP ${response.status}`);
        }
        _currentStreamingMessage = _displayAiMessage('', false, null, false);
        _currentStreamingText = '';
        const markdownBody = _currentStreamingMessage.querySelector('.content .markdown-body');
        if (!markdownBody) throw new Error('UI error');

        let firstTokenReceived = false, streamFinished = false, searchResults = null;
        const reader = response.body.getReader();
        _currentStreamReader = reader;
        const decoder = new TextDecoder();
        let buffer = '';

        while (!streamFinished) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (dataStr === '[DONE]') { streamFinished = true; break; }

                try {
                    const data = JSON.parse(dataStr);
                    if (data.status === 'searching') {
                        _showAiTypingIndicator(true, `🔍 Ищу: ${data.query || '…'}`);
                        continue;
                    }
                    if (data.status === 'search_done') {
                        _showAiTypingIndicator(true, '📄 Загружаю содержимое страниц…');
                        continue;
                    }
                    if (data.status === 'search_error') {
                        _showAiTypingIndicator(true, '⚠️ Поиск не удался, отвечаю из памяти…');
                        continue;
                    }
                    if (data.status === 'fetching_pages') {
                        _showAiTypingIndicator(true, `📖 Читаю ${data.count || 'несколько'} страниц…`);
                        continue;
                    }
                    if (data.token) {
                        if (!firstTokenReceived) {
                            _showAiTypingIndicator(false);
                            firstTokenReceived = true;
                        }
                        _currentStreamingText += data.token;

                        // Обновляем DOM с debounce (не чаще 50 мс)
                        if (!window._aiUpdateTimer) {
                            window._aiUpdateTimer = setTimeout(() => {
                                markdownBody.innerHTML = _renderMarkdown(_currentStreamingText);
                                _enhanceCodeBlocks(markdownBody);
                                _addImageDownloadButtons(markdownBody);
                                _attachReasoningToggle(markdownBody);
                                if (_aiMessagesContainer) _aiMessagesContainer.scrollTop = _aiMessagesContainer.scrollHeight;
                                window._aiUpdateTimer = null;
                            }, 50);
                        }
                    } else if (data.error) {
                        markdownBody.textContent = '❌ ' + data.error;
                        firstTokenReceived = true;
                        streamFinished = true;
                        break;
                    } else if (data.sources) {
                        searchResults = data.sources;
                    }
                } catch(e) {}
            }
        }

        // Принудительно обновляем после завершения стрима (если есть таймер)
        if (window._aiUpdateTimer) {
            clearTimeout(window._aiUpdateTimer);
            window._aiUpdateTimer = null;
        }

        if (!firstTokenReceived) {
            _showAiTypingIndicator(false);
            if (markdownBody) markdownBody.textContent = '🤖 Нет ответа от модели.';
        } else if (_currentStreamingText) {
            const finalHtml = _renderMarkdown(_currentStreamingText);
            markdownBody.innerHTML = finalHtml;
            _enhanceCodeBlocks(markdownBody);
            _addImageDownloadButtons(markdownBody);
            _attachReasoningToggle(markdownBody);
            _saveAiMessage('assistant', _currentStreamingText, _currentAiSessionId);
            if (searchResults && searchResults.length) {
                _displaySearchSources(searchResults);
            } else if (useWebSearch && !searchResults) {
                _tryFetchSearchSources(messageText);
            }
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            _showAiTypingIndicator(false);
            if (_currentStreamingMessage?.parentNode) {
                const errDiv = _currentStreamingMessage.querySelector('.content .markdown-body');
                if (errDiv) errDiv.textContent = '⏹ Ответ прерван пользователем.';
            } else {
                _displayAiMessage('⏹ Ответ прерван.', false, null, true);
            }
            _showToast('Генерация остановлена', 'warning');
        } else {
            console.error('AI error:', err);
            _showAiTypingIndicator(false);
            if (_currentStreamingMessage?.parentNode) {
                const errDiv = _currentStreamingMessage.querySelector('.content .markdown-body');
                if (errDiv) errDiv.textContent = '❌ Ошибка связи с AI-сервером. Проверьте, запущен ли LM Studio.';
            } else {
                _displayAiMessage('❌ Ошибка связи с AI-сервером.', false, null, true);
            }
        }
    } finally {
        if (_currentStreamReader) {
            try { _currentStreamReader.releaseLock(); } catch(e) {}
            _currentStreamReader = null;
        }
        _currentStreamingMessage = null;
        _currentStreamingText = '';
        _isSending = false;
        _currentAbortController = null;

        if (_aiSendBtn) _aiSendBtn.style.display = 'inline-flex';
        if (_aiStopBtn) _aiStopBtn.style.display = 'none';
    }
}

    async function _tryFetchSearchSources(query) {
        try {
            const res = await fetch(CONFIG.searchEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            if (res.ok) {
                const data = await res.json();
                if (data.results?.length) _displaySearchSources(data.results);
            }
        } catch(e) {}
    }

    // ─── превью изображения ───
    function _showImagePreview(file) {
        const oldPreview = document.getElementById('aiImagePreview');
        if (oldPreview) oldPreview.remove();
        const previewContainer = document.createElement('div');
        previewContainer.id = 'aiImagePreview';
        previewContainer.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            margin: 8px 16px 0 16px;
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            border: 1px solid var(--border-color);
        `;
        const blobUrl = URL.createObjectURL(file);
        if (_currentImagePreviewUrl) URL.revokeObjectURL(_currentImagePreviewUrl);
        _currentImagePreviewUrl = blobUrl;
        previewContainer.innerHTML = `
            <img src="${blobUrl}" style="width: 40px; height: 40px; object-fit: cover; border-radius: 6px;">
            <span style="font-size: 13px; color: var(--text-secondary); flex: 1;">${_escapeHtml(file.name)}</span>
            <button type="button" id="clearAiImage" class="btn btn-icon" style="font-size: 16px; padding: 4px;">✕</button>
        `;
        const form = document.querySelector('#aiChatContainer .input-area');
        if (form) {
            form.insertBefore(previewContainer, form.firstChild);
        } else {
            const container = document.getElementById('aiChatContainer');
            if (container) container.appendChild(previewContainer);
        }
        document.getElementById('clearAiImage')?.addEventListener('click', () => {
            if (_currentImagePreviewUrl) URL.revokeObjectURL(_currentImagePreviewUrl);
            _pendingImageFile = null;
            previewContainer.remove();
            _currentImagePreviewUrl = null;
        });
    }
    function _clearImagePreview() {
        const previewDiv = document.getElementById('aiImagePreview');
        if (previewDiv) previewDiv.remove();
        if (_currentImagePreviewUrl) {
            URL.revokeObjectURL(_currentImagePreviewUrl);
            _currentImagePreviewUrl = null;
        }
        _pendingImageFile = null;
    }

    // ─── остановка генерации ───
    function _stopAiGeneration() {
    if (_currentAbortController) {
        _currentAbortController.abort();
        _currentAbortController = null;
        // ----- ДОБАВИТЬ ВОССТАНОВЛЕНИЕ -----
        if (_aiStopBtn) _aiStopBtn.style.display = 'none';
        if (_aiSendBtn) _aiSendBtn.style.display = 'inline-flex';
        // ------------------------------------
        _showToast('Остановка генерации...', 'warning');
    } else {
        _showToast('Нет активного запроса', 'warning');
    }
}

    // ─── обновление стилей кнопок ───
    function _updateInternetBtnStyle() {
        if (!_aiInternetBtn) return;
        _aiInternetBtn.setAttribute('data-active', _internetEnabled ? 'true' : 'false');
        _aiInternetBtn.title = _internetEnabled
            ? 'Интернет ВКЛЮЧЁН (буду искать и читать страницы)'
            : 'Интернет ВЫКЛЮЧЕН (только мои знания)';
    }
    function _updateReasoningBtnStyle() {
        if (!_aiReasoningBtn) return;
        _aiReasoningBtn.setAttribute('data-active', _reasoningEnabled ? 'true' : 'false');
        _aiReasoningBtn.title = _reasoningEnabled ? 'Reasoning ON' : 'Reasoning OFF';
    }

    // ─── настройка UI ───
    function _setupAiUI() {
        if (!_aiSendBtn) return;

        if (_aiNewChatBtn) {
            _aiNewChatBtn.onclick = () => {
                if (typeof createAiSession === 'function') {
                    const id = createAiSession();
                    loadAiSessionsIntoConversations();
                    if (typeof window.selectConversation === 'function') {
                        const sessions = getAiSessions();
                        const found = sessions.find(s => s.id === id);
                        window.selectConversation(id, found ? found.name : 'AI Chat', false);
                    }
                }
            };
        }
        if (_aiStopBtn) {
            _aiStopBtn.onclick = _stopAiGeneration;
            _aiStopBtn.style.display = 'none';
        }

        _aiSendBtn.onclick = async () => {
            const text = _aiMessageInput ? _aiMessageInput.value.trim() : '';
            const image = _pendingImageFile;
            if (!text && !image) {
                _showToast('Введите сообщение или прикрепите изображение', 'warning');
                return;
            }
            _aiMessageInput.disabled = true;
            _aiSendBtn.disabled = true;
            _aiAttachBtn.disabled = true;
            if (_aiImageGenBtn) _aiImageGenBtn.disabled = true;
            if (_aiInternetBtn) _aiInternetBtn.disabled = true;
            if (_aiReasoningBtn) _aiReasoningBtn.disabled = true;
            if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = true;

            const originalText = text;
            const imageToSend = image;
            if (_aiMessageInput) _aiMessageInput.value = '';
            _clearImagePreview();
            try {
                await _sendToAi(originalText, imageToSend);
            } finally {
                _aiMessageInput.disabled = false;
                _aiSendBtn.disabled = false;
                _aiAttachBtn.disabled = false;
                if (_aiImageGenBtn) _aiImageGenBtn.disabled = false;
                if (_aiInternetBtn) _aiInternetBtn.disabled = false;
                if (_aiReasoningBtn) _aiReasoningBtn.disabled = false;
                if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = false;
                _aiMessageInput.focus();
            }
        };

        if (_aiMessageInput) {
            _aiMessageInput.onkeydown = (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    _aiSendBtn?.click();
                }
            };
        }

        if (_aiAttachBtn && _aiImageInput) {
            _aiAttachBtn.onclick = () => _aiImageInput.click();
            _aiImageInput.onchange = async (e) => {
                const file = e.target.files[0];
                if (file && file.type.startsWith('image/')) {
                    _pendingImageFile = file;
                    _showImagePreview(file);
                } else {
                    _showToast('Пожалуйста, выберите изображение', 'warning');
                }
                _aiImageInput.value = '';
            };
        }

        if (_aiReasoningBtn) {
            _reasoningEnabled = localStorage.getItem('ai_reasoning_mode') === 'true';
            _updateReasoningBtnStyle();
            _aiReasoningBtn.onclick = () => {
                _reasoningEnabled = !_reasoningEnabled;
                _updateReasoningBtnStyle();
                localStorage.setItem('ai_reasoning_mode', _reasoningEnabled);
                _showToast(`Режим рассуждений ${_reasoningEnabled ? 'включён 🧠' : 'выключен'}`, 'info');
            };
        }

        if (_aiInternetBtn) {
            _internetEnabled = localStorage.getItem('ai_internet') === 'true';
            _updateInternetBtnStyle();
            _aiInternetBtn.onclick = () => {
                _internetEnabled = !_internetEnabled;
                localStorage.setItem('ai_internet', _internetEnabled);
                _updateInternetBtnStyle();
                _showToast(`Интернет-поиск ${_internetEnabled ? 'включён 🌐 (буду загружать страницы)' : 'выключен'}`, 'info');
            };
        }

        if (_aiClearHistoryBtn) _aiClearHistoryBtn.onclick = _clearAiHistory;

        if (_closeAiChatBtn) {
            _closeAiChatBtn.onclick = () => {
                const firstConv = document.querySelector('.conversation-item:not([data-address^="ai_"])');
                if (firstConv?.dataset.address) {
                    window.selectConversation(firstConv.dataset.address, '', firstConv.dataset.isGroup === '1');
                } else {
                    window.selectConversation('', '', false);
                }
            };
        }

        // Генерация изображений
        if (_aiImageGenBtn) {
            _aiImageGenBtn.onclick = async () => {
                const rawPrompt = _aiMessageInput.value.trim();
                if (!rawPrompt) {
                    _showToast('Enter a prompt first', 'warning');
                    return;
                }
                if (_isSending) {
                    _showToast('Please wait, current request in progress', 'warning');
                    return;
                }
                _isSending = true;
                if (_aiMessageInput) _aiMessageInput.value = '';
                _aiMessageInput.disabled = true;
                _aiSendBtn.disabled = true;
                _aiAttachBtn.disabled = true;
                _aiImageGenBtn.disabled = true;
                if (_aiInternetBtn) _aiInternetBtn.disabled = true;
                if (_aiReasoningBtn) _aiReasoningBtn.disabled = true;
                if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = true;

                _showAiTypingIndicator(true, '🎨 Generating image...');

                try {
                    let finalPrompt = rawPrompt;
                    _showAiTypingIndicator(true, '✨ Enhancing prompt with AI...');
                    try {
                        const enhanceResp = await fetch('/ai/enhance_prompt', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ prompt: rawPrompt })
                        });
                        if (enhanceResp.ok) {
                            const enhanceData = await enhanceResp.json();
                            if (enhanceData.enhanced && enhanceData.enhanced !== rawPrompt) {
                                finalPrompt = enhanceData.enhanced;
                                _showToast('Prompt enhanced', 'success');
                            }
                        }
                    } catch (enhanceErr) {
                        console.warn('Enhance failed, using original prompt', enhanceErr);
                    }
                    _showAiTypingIndicator(true, '🎨 Generating image...');

                    const response = await fetch('/ai/generate_image', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: finalPrompt })
                    });
                    const data = await response.json();
                    if (response.ok && data.image_base64) {
                        const imageMarkdown = `![generated](data:image/png;base64,${data.image_base64})`;
                        const finalText = `🎨 *Generated image for:*\n> ${rawPrompt}\n\n${imageMarkdown}`;
                        _displayAiMessage(finalText, false, null, true);
                    } else {
                        _showToast(data.detail || data.error || 'Generation failed', 'error');
                    }
                } catch (err) {
                    console.error('Image generation error:', err);
                    _showToast('Network error or service unavailable', 'error');
                } finally {
                    _isSending = false;
                    _showAiTypingIndicator(false);
                    _aiMessageInput.disabled = false;
                    _aiSendBtn.disabled = false;
                    _aiAttachBtn.disabled = false;
                    _aiImageGenBtn.disabled = false;
                    if (_aiInternetBtn) _aiInternetBtn.disabled = false;
                    if (_aiReasoningBtn) _aiReasoningBtn.disabled = false;
                    if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = false;
                    _aiMessageInput.focus();
                }
            };
        }

        // Исследование
        const researchBtn = document.getElementById('aiResearchBtn');
        if (researchBtn) {
            researchBtn.onclick = async () => {
                const text = _aiMessageInput.value.trim();
                if (!text) {
                    _showToast('Введите вопрос для исследования', 'warning');
                    return;
                }
                if (_isSending) {
                    _showToast('Подождите, текущий запрос обрабатывается', 'warning');
                    return;
                }
                _isSending = true;
                const originalIcon = researchBtn.textContent;
                researchBtn.textContent = '⏳';
                researchBtn.setAttribute('data-active', 'true');

                _aiMessageInput.disabled = true;
                _aiSendBtn.disabled = true;
                _aiAttachBtn.disabled = true;
                if (_aiImageGenBtn) _aiImageGenBtn.disabled = true;
                if (_aiInternetBtn) _aiInternetBtn.disabled = true;
                if (_aiReasoningBtn) _aiReasoningBtn.disabled = true;
                if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = true;

                const originalText = text;
                _aiMessageInput.value = '';
                _clearImagePreview();
                _displayAiMessage(originalText, true, null, true);
                try {
                    await _sendResearchQuery(originalText);
                } finally {
                    _isSending = false;
                    researchBtn.textContent = originalIcon;
                    researchBtn.setAttribute('data-active', 'false');
                    _aiMessageInput.disabled = false;
                    _aiSendBtn.disabled = false;
                    _aiAttachBtn.disabled = false;
                    if (_aiImageGenBtn) _aiImageGenBtn.disabled = false;
                    if (_aiInternetBtn) _aiInternetBtn.disabled = false;
                    if (_aiReasoningBtn) _aiReasoningBtn.disabled = false;
                    if (_aiClearHistoryBtn) _aiClearHistoryBtn.disabled = false;
                    _aiMessageInput.focus();
                }
            };
        }
    }

    // ─── инициализация сессии ───
    function _initAiChatSession(sessionId) {
        if (_aiChatActive && _currentAiSessionId === sessionId) return;
        _aiChatActive = true;
        _currentAiSessionId = sessionId || null;
        _aiNameSet = false;

        _aiMessagesContainer = document.getElementById('aiMessagesContainer');
        _aiMessageInput = document.getElementById('aiMessageInput');
        _aiSendBtn = document.getElementById('aiSendBtn');
        _aiAttachBtn = document.getElementById('aiAttachBtn');
        _aiReasoningBtn = document.getElementById('aiReasoningBtn');
        _aiInternetBtn = document.getElementById('aiInternetBtn');
        _aiImageInput = document.getElementById('aiImageInput');
        _aiClearHistoryBtn = document.getElementById('aiClearHistoryBtn');
        _closeAiChatBtn = document.getElementById('closeAiChatBtn');
        _aiStopBtn = document.getElementById('aiStopBtn');
        _aiNewChatBtn = document.getElementById('aiNewChatBtn');
        _aiImageGenBtn = document.getElementById('aiImageGenBtn');
        if (!_aiMessagesContainer) return;
        _loadAiHistory(_currentAiSessionId);
        _setupAiUI();
        if (_aiStopBtn) _aiStopBtn.style.display = 'none';
    }

    // ─── экспорт ───
    window._initAiChatSession = _initAiChatSession;
    window.createAiSession = createAiSession;
    window.getAiSessions = getAiSessions;
    window.saveAiSessions = saveAiSessions;
    window.updateAiSessionName = updateAiSessionName;
    window.loadAiSessionsIntoConversations = loadAiSessionsIntoConversations;

    // =================================================================
    // 4. ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
    // =================================================================
    document.addEventListener('DOMContentLoaded', function() {
        // Загружаем AI-сессии в список разговоров
        loadAiSessionsIntoConversations();

        // Добавляем кнопку "Новый AI чат" в панель бесед, если её нет
        const headerActions = document.querySelector('.conversations-header .header-actions');
        if (headerActions) {
            const existingBtn = document.getElementById('newAiChatBtn');
            if (!existingBtn) {
                const btn = document.createElement('button');
                btn.id = 'newAiChatBtn';
                btn.className = 'btn-icon-oval';
                btn.title = 'New AI chat';
                btn.innerHTML = '🤖+';
                btn.onclick = function() {
                    const id = createAiSession();
                    loadAiSessionsIntoConversations();
                    const sessions = getAiSessions();
                    const found = sessions.find(s => s.id === id);
                    window.selectConversation(id, found ? found.name : 'AI Chat', false);
                };
                headerActions.appendChild(btn);
            }
        }

        // Если текущий выбранный чат — AI, инициализируем его
        const currentAddr = window.State?.currentChatAddress;
        if (currentAddr && currentAddr.startsWith('ai_')) {
            const sessions = getAiSessions();
            const found = sessions.find(s => s.id === currentAddr);
            window.selectConversation(currentAddr, found ? found.name : 'AI Chat', false);
        }
    });

    // Перехватываем добавление новых AI-сессий в список при вызове loadConversations
    const _origLoadConversations = window.loadConversations;
    if (_origLoadConversations) {
        window.loadConversations = function() {
            const result = _origLoadConversations.apply(this, arguments);
            setTimeout(loadAiSessionsIntoConversations, 50);
            return result;
        };
    }

    console.log('✅ AI Manager loaded — multiple AI sessions with auto-naming enabled');
})();