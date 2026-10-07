    // ========== КОНТЕКСТНОЕ МЕНЮ БЕСЕД (ПКМ + долгое нажатие) ==========
    const CONV_ICONS = {
        open:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
        call:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/></svg>',
        user:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        copy:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
        trash:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
    };

    function hiddenChatsKey() {
        return 'hidden_chats_' + ((window.State && State.userAddress) || 'anon');
    }
    function getHiddenChats() {
        try { return JSON.parse(localStorage.getItem(hiddenChatsKey()) || '[]'); } catch (e) { return []; }
    }
    window.isChatHidden = function (address) {
        return getHiddenChats().some(a => String(a).toLowerCase() === String(address).toLowerCase());
    };

    async function hideConversation(address, name) {
        const confirmed = await window.showConfirmModal(t('hide_chat'), t('confirm_hide_chat', { name: name || '' }));
        if (!confirmed) return;
        try {
            const res = await fetch('/hide_conversation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chat_with: address })
            });
            if (!res.ok) throw new Error('HTTP ' + res.status);
        } catch (err) {
            console.warn('hide_conversation server error, hiding locally:', err);
        }
        const key = hiddenChatsKey();
        const list = getHiddenChats();
        if (!list.some(a => String(a).toLowerCase() === String(address).toLowerCase())) list.push(address);
        localStorage.setItem(key, JSON.stringify(list));
        const el = document.querySelector(`.conversation-item[data-address="${CSS.escape(address)}"]`);
        if (el) el.remove();
        if (window.State && State.currentChatAddress === address) {
            State.currentChatAddress = null;
            const panel = document.getElementById('chatPanel');
            if (panel) panel.classList.remove('open');
        }
        window.NotificationManager?.showToast(t('chat_hidden'), 'success');
    }

    function bindConversationHold(item, address, displayName, isGroup) {
        if (!window.ContextMenu || !address || address === 'ai_bot') return;
        ContextMenu.bind(item, () => {
            const items = [
                { icon: CONV_ICONS.open, label: t('open_chat'), onClick: () => window.selectConversation(address, displayName, isGroup) },
            ];
            if (!isGroup) {
                items.push(
                    { icon: CONV_ICONS.call, label: t('call'), onClick: () => {
                        if (window.CallManager && typeof window.CallManager.makeCall === 'function') {
                            window.CallManager.makeCall(address, false, displayName);
                        } else {
                            window.NotificationManager?.showToast(t('calls_not_available'), 'warning');
                        }
                    } },
                    { icon: CONV_ICONS.user, label: t('add_to_contacts'), onClick: () => {
                        window.location.href = '/contacts?start_with=' + encodeURIComponent(address) + '&name=' + encodeURIComponent(displayName);
                    } }
                );
            }
            items.push(
                { icon: CONV_ICONS.copy, label: t('copy_address'), onClick: () => {
                    const done = () => window.NotificationManager?.showToast(t('copied_to_clipboard'), 'success');
                    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(address).then(done).catch(done);
                    else done();
                } },
                { separator: true },
                { icon: CONV_ICONS.trash, label: t('delete_chat'), danger: true, onClick: () => hideConversation(address, displayName) }
            );
            return items;
        });
    }
    window.bindConversationHold = bindConversationHold;

    // ========== ДИНАМИЧЕСКИЙ ОТСТУП ПОД ПОЛЕ ВВОДА ==========
    window.adjustMessagesPadding = function() {
        const aiContainer = document.getElementById('aiChatContainer');
        const isAiActive = aiContainer && !aiContainer.classList.contains('hidden');
        let messages, inputArea;

        if (isAiActive) {
            messages = document.getElementById('aiMessagesContainer');
            inputArea = aiContainer.querySelector('.input-area');
        } else {
            messages = document.getElementById('messagesContainer');
            const chatPanel = document.querySelector('.chat-panel');
            inputArea = chatPanel ? chatPanel.querySelector('.input-area') : null;
        }

        if (messages && inputArea) {
            const height = inputArea.offsetHeight;
            messages.style.paddingBottom = (height + 20) + 'px';
        }
    };

    async function selectConversation(address, name, isGroup) {
        if (State.topObserver) { State.topObserver.disconnect(); State.topObserver = null; }
        if (window.isSending) window.isSending = false;
        State.currentChatAddress = address;
        State.currentChatIsGroup = !!isGroup;
        State.currentChatPartnerAddress = isGroup ? '' : (address === State.userAddress ? '' : address);

        const callBtn = document.getElementById('callButton');
        if (callBtn) {
            if (!isGroup && address && address !== State.userAddress && address !== 'ai_bot') {
                callBtn.style.display = 'inline-flex';
                callBtn.onclick = () => window.CallManager?.makeCall(address, false);
            } else {
                callBtn.style.display = 'none';
            }
        }

        if (window.clearMainImagePreview) window.clearMainImagePreview();
        else { const previewDiv = document.getElementById('mainImagePreview'); if (previewDiv) previewDiv.remove(); }

        fetch('/heartbeat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ current_chat: address }) }).catch(e=>{});

        const aiContainer = document.getElementById('aiChatContainer');
        const mainContainer = document.getElementById('messagesContainer');
        const mainInputArea = document.querySelector('.chat-panel .input-area');
        const mainChatHeader = document.querySelector('.chat-panel .chat-panel-header');

        if (address === 'ai_bot') {
            if (mainContainer) mainContainer.style.display = 'none';
            if (mainInputArea) mainInputArea.style.display = 'none';
            if (mainChatHeader) mainChatHeader.style.display = 'none';
            if (aiContainer) aiContainer.classList.remove('hidden');
            if (typeof window.initAiChat === 'function') window.initAiChat();
            document.getElementById('currentChatName').textContent = t('ai_assistant');
            document.getElementById('chatSubtitle').textContent = t('streaming_response');
            _enableChatControls();
            document.querySelectorAll('.conversation-item').forEach(item => item.classList.remove('active'));
            return;
        } else {
            if (aiContainer) aiContainer.classList.add('hidden');
            if (mainContainer) mainContainer.style.display = '';
            if (mainInputArea) mainInputArea.style.display = '';
            if (mainChatHeader) mainChatHeader.style.display = '';
        }

        if (isGroup) {
            try {
                const res = await fetch('/get_groups');
                const data = await res.json();
                const group = data.groups.find(g => 'group:' + g.id === address);
                State.currentGroupMembers = group ? group.members : [];
            } catch(e) { State.currentGroupMembers = []; }
        } else State.currentGroupMembers = null;

        if (window.NotificationManager?.setActiveChat) window.NotificationManager.setActiveChat(address);
        const container = document.getElementById('messagesContainer');
        if (container) { container.innerHTML = `<div class="loading">${t('loading')}</div>`; container.classList.add('loading'); }
        _disableChatControls();
        const nameEl = document.getElementById('currentChatName');
        if (nameEl) nameEl.textContent = name || t('loading');
        const subtitleEl = document.getElementById('chatSubtitle');
        if (subtitleEl) subtitleEl.textContent = isGroup ? t('group_chat') : t('direct_message');
        document.querySelectorAll('.conversation-item').forEach(item => item.classList.toggle('active', item.dataset.address === address));
        State.lastKnownMessageId = 0;
        State.lastMessageTimestamp = 0;
        State.pendingImageData = null;
       if (window.stopStatusPolling) window.stopStatusPolling();
       if (window.startStatusPolling) window.startStatusPolling();
       if (window.stopUserStatusPolling) window.stopUserStatusPolling();
       if (window.startUserStatusPolling) window.startUserStatusPolling();

        if (container) {
            container.querySelectorAll('[data-object-url]').forEach(el => {
               URL.revokeObjectURL(el.dataset.objectUrl);
            });
        }
        await loadMessagesForConversation(address, false);
    }

    async function loadMessagesForConversation(chatWithAddress, isNewMessage = false, forceScroll = false) {
        const container = document.getElementById('messagesContainer');
        if (!container) return;
        if (!chatWithAddress) {
            if (!isNewMessage) container.innerHTML = `<div class="empty-state animate-fade"><div class="icon">💬</div><p>${t('select_conversation_to_start')}</p></div>`;
            _enableChatControls();
            return;
        }

        const cached = window.getCachedMessages(chatWithAddress);
        let lastKnownId = 0;

        if (!isNewMessage && cached.length > 0) {
            container.querySelectorAll('[data-object-url]').forEach(el => {
            URL.revokeObjectURL(el.dataset.objectUrl);
            });
            container.innerHTML = '';
            const decryptedCache = [];
            for (const msg of cached) {
                let displayMsg = msg;
                if (!msg.isDecrypted) {
                    try {
                        displayMsg = await window.processMessageDecryption(msg);
                    } catch(e) {
                        displayMsg = { ...msg, content: '🔒 Decrypt error', isDecrypted: false };
                    }
                }
                decryptedCache.push(displayMsg);
                if (displayMsg.id > lastKnownId) lastKnownId = displayMsg.id;
            }
            renderMessagesWithSeparators(container, decryptedCache);
            const wasAtBottom = isUserAtBottom(container, 30);
            if (wasAtBottom || forceScroll) {
                container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
            } else {
                showNewMessagesBadge();
            }
            State.lastKnownMessageId = lastKnownId;
            _enableChatControls();
            setupTopObserver();
            if (cached.length && (wasAtBottom || forceScroll)) {
                markConversationAsRead(chatWithAddress, cached[cached.length-1].id);
            }
            adjustMessagesPadding();
        } else if (!isNewMessage) {
            container.innerHTML = `<div class="loading">${t('loading_messages')}</div>`;
            container.classList.add('loading');
        }

        try {
            const params = new URLSearchParams({ with: chatWithAddress });
            if (lastKnownId > 0) params.append('last_message_id', lastKnownId);
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 10000);
            const res = await fetch('/get_conversation?' + params.toString(), { signal: controller.signal });
            clearTimeout(timeout);

            if (res.status === 403) {
                window.NotificationManager?.showToast(t('chat_unavailable_group_deleted'), 'error');
                const convItem = document.querySelector(`.conversation-item[data-address="${chatWithAddress}"]`);
                if (convItem) convItem.remove();
                if (State.currentChatAddress === chatWithAddress) {
                    container.innerHTML = `<div class="empty-state"><p>${t('chat_unavailable')}</p></div>`;
                    State.currentChatAddress = '';
                    _enableChatControls();
                }
                return;
            }
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load');

            const rawMessages = Array.isArray(data.messages) ? data.messages : [];
            if (rawMessages.length === 0) {
                if (!isNewMessage && cached.length === 0) {
                    container.innerHTML = `<div class="empty-state animate-fade"><div class="icon">👋</div><p>${t('no_messages_yet')}</p><p class="text-muted" style="font-size:12px">${t('start_conversation')}</p></div>`;
                    _enableChatControls();
                }
                container.classList.remove('loading');
                adjustMessagesPadding();
                return;
            }

            const newMessages = [];
            for (const msg of rawMessages) {
                try {
                    const decrypted = await window.processMessageDecryption(msg);
                    newMessages.push(decrypted);
                } catch(e) {
                    newMessages.push({ ...msg, content: '🔒 Decrypt error', image: null });
                }
            }
            window.addMessagesToCache(chatWithAddress, newMessages, 'end');

            if (container) {
                const existingIds = new Set(Array.from(container.querySelectorAll('.message')).map(el => el.dataset.messageId));
                const uniqueNew = newMessages.filter(msg => !existingIds.has(String(msg.id)));
                if (uniqueNew.length) {
                    renderMessagesWithSeparators(container, uniqueNew);
                    if (uniqueNew.some(m => m.id > State.lastKnownMessageId))
                        State.lastKnownMessageId = Math.max(State.lastKnownMessageId, ...uniqueNew.map(m => m.id));
                }
                const wasAtBottom = isUserAtBottom(container, 30);
                const isFirstOpen = !isNewMessage && cached.length === 0;
                if (wasAtBottom || isFirstOpen || forceScroll) {
                    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
                } else if (uniqueNew.length && !isNewMessage) {
                    showNewMessagesBadge();
                }

                if ((wasAtBottom || forceScroll) && newMessages.length) {
                    markConversationAsRead(chatWithAddress, newMessages[newMessages.length-1].id);
                }
                adjustMessagesPadding();
            }

            if (!isNewMessage && cached.length === 0 && newMessages.length) setupTopObserver();
            _enableChatControls();

            setTimeout(async () => {
                const myMessages = document.querySelectorAll('.message-own');
                const ids = Array.from(myMessages).filter(el => el.dataset.id && !el.dataset.id.startsWith('temp')).map(el => el.dataset.id);
                if (ids.length) {
                    try {
                        const statusRes = await fetch('/message/statuses', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ ids })
                        });
                        const statuses = await statusRes.json();
                        for (const [id, st] of Object.entries(statuses)) {
                            const msgDiv = document.querySelector(`.message-own[data-id="${id}"]`);
                            if (msgDiv && msgDiv.dataset.status !== st) {
                                msgDiv.dataset.status = st;
                                updateStatusIcon(msgDiv, st);
                            }
                        }
                    } catch(e) { console.warn('Status refresh error', e); }
                }
            }, 100);
        } catch (error) {
            console.error('Load messages error:', error);
            container.classList.remove('loading');
            if (!isNewMessage && cached.length === 0) container.innerHTML = `<p class="text-muted text-center">${t('failed_to_load_messages')}</p>`;
            _enableChatControls();
        }
        container.classList.remove('loading');
    }

    function markConversationAsRead(chatId, explicitLastMessageId) {
        const item = document.querySelector(`.conversation-item[data-address="${chatId}"]`);
        if (item) { const meta = item.querySelector('.meta .truncate'); if (meta) { meta.textContent = t('read_status'); meta.style.fontStyle = 'italic'; } }
        let lastMessageId = explicitLastMessageId;
        if (lastMessageId === undefined) {
            const lastMsg = document.querySelector('#messagesContainer .message:last-of-type');
            lastMessageId = lastMsg ? parseInt(lastMsg.dataset.messageId) : 0;
        }
        fetch('/mark_conversation_read', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ chat_with: chatId, last_message_id: lastMessageId }) }).catch(e=>console.debug);
    }

    function updateConversationPreview(chatId, newPreview) {
        const items = document.querySelectorAll('.conversation-item');
        for (const item of items) {
            if (item.dataset.address === chatId) { const meta = item.querySelector('.meta .truncate'); if (meta) meta.textContent = newPreview; break; }
        }
    }

    function _disableChatControls() {
        ['messageContent', 'attachImageButton', 'attachAudioButton', 'recordAudioButton', 'sendButton', 'addToContactsBtn', 'clearConversationBtn'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = true; });
    }

    function _enableChatControls() {
        ['messageContent', 'attachImageButton', 'attachAudioButton', 'recordAudioButton', 'sendButton', 'addToContactsBtn', 'clearConversationBtn'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = false; });
        const btn = document.getElementById('addToContactsBtn');
        if (btn) {
            if (State.currentChatIsGroup || !State.currentChatPartnerAddress || State.currentChatPartnerAddress === State.userAddress) {
                btn.disabled = true;
                btn.title = t('cannot_add_group_or_self');
            } else {
                btn.disabled = false;
                btn.title = t('add_to_contacts');
            }
        }
        const messageInput = document.getElementById('messageContent');
        if (messageInput && !window._pushRequested) {
    messageInput.addEventListener('focus', async () => {
        if (Notification.permission === 'default' && window.NotificationManager?.requestNotificationPermission) {
            window._pushRequested = true;
            await window.NotificationManager.requestNotificationPermission();
        }
    }, { once: true });
}
    }

    function openImageModal(imageUrl) {
        const modal = document.getElementById('imageModal');
        const img = document.getElementById('modalImage');
        if (!modal || !img) return;
        img.src = imageUrl;
        modal.classList.remove('hidden');
        const downloadBtn = document.getElementById('downloadImageBtn');
        if (downloadBtn) {
            const newBtn = downloadBtn.cloneNode(true);
            downloadBtn.parentNode.replaceChild(newBtn, downloadBtn);
            newBtn.onclick = () => {
                const a = document.createElement('a'); a.href = img.src; a.download = 'image.png'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
                if (window.NotificationManager) window.NotificationManager.showToast(t('image_saved'), 'success');
            };
        }
    }

    function closeImageModal() { const modal = document.getElementById('imageModal'); if (modal) modal.classList.add('hidden'); }

    window.onNewMessageReceived = function(decrypted) {
        const container = document.getElementById('messagesContainer');
        if (container) {
            const wasAtBottom = isUserAtBottom(container, 30);
            if (document.getElementById('msg-' + decrypted.id)) return;

            const lastMsg = container.querySelector('.message:last-of-type');
            let lastTimestamp = null;
            if (lastMsg) {
                const lastMsgId = lastMsg.dataset.messageId;
                const cachedMsgs = window.getCachedMessages(State.currentChatAddress);
                const lastMsgObj = cachedMsgs?.find(m => m.id == lastMsgId);
                if (lastMsgObj) lastTimestamp = lastMsgObj.timestamp;
            }
            const currentDate = new Date(decrypted.timestamp * 1000).toDateString();
            const lastDate = lastTimestamp ? new Date(lastTimestamp * 1000).toDateString() : null;
            if (currentDate !== lastDate) {
                const divider = document.createElement('div');
                divider.className = 'date-divider';
                divider.textContent = formatDateDivider(decrypted.timestamp);
                container.appendChild(divider);
            }

            const msgElement = createMessageElement(decrypted);
            container.appendChild(msgElement);
            adjustMessagesPadding();

            if (wasAtBottom) {
                setTimeout(() => container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' }), 50);
            } else {
                showNewMessagesBadge();
            }

            if (!decrypted.is_mine && wasAtBottom) {
                fetch(`/message/${decrypted.id}/read`, { method: 'POST' }).catch(e => console.warn(e));
                markConversationAsRead(decrypted.chatId || State.currentChatAddress, decrypted.id);
            }
        }
    };

    window.updateConversationStatus = function(address, status) {
        const item = document.querySelector(`.conversation-item[data-address="${address}"]`);
        if (item && !item.dataset.isGroup) {
            const statusSpan = item.querySelector('.status');
            if (statusSpan) {
                statusSpan.className = `status ${status}`;
                statusSpan.title = status === 'online' ? t('online') : t('offline');
            }
        }
    };

    window.moveConversationToTop = function(chatId) {
    const container = document.querySelector('.conversations-list');
    if (!container) return;
    const item = document.querySelector(`.conversation-item[data-address="${chatId}"]`);
    if (item && item.parentNode === container) {
        container.insertBefore(item, container.firstChild);
        item.classList.add('new-message-highlight');
        setTimeout(() => item.classList.remove('new-message-highlight'), 500);
    }
};

    window.loadConversations = loadConversations;
    window.selectConversation = selectConversation;
    window.loadMessagesForConversation = loadMessagesForConversation;
    window.createMessageElement = createMessageElement;
    window.updateStatusIcon = updateStatusIcon;
    window.updateConversationPreview = updateConversationPreview;
    window.markConversationAsRead = markConversationAsRead;
    window.smartScrollToBottom = smartScrollToBottom;
    window.setupTopObserver = setupTopObserver;
    window._enableChatControls = _enableChatControls;
    window._disableChatControls = _disableChatControls;
    window.fetchUserStatuses = fetchUserStatuses;
    window.openImageModal = openImageModal;
    window.closeImageModal = closeImageModal;
    window.adjustMessagesPadding = adjustMessagesPadding;

    // ========== ИНИЦИАЛИЗАЦИЯ ==========
    document.addEventListener('DOMContentLoaded', () => {
        const inputArea = document.querySelector('.chat-panel .input-area');
        if (inputArea && window.ResizeObserver) {
            const resizeObserver = new ResizeObserver(() => adjustMessagesPadding());
            resizeObserver.observe(inputArea);
        }

        const aiInputArea = document.querySelector('#aiChatContainer .input-area');
        if (aiInputArea && window.ResizeObserver) {
            const aiResizeObserver = new ResizeObserver(() => adjustMessagesPadding());
            aiResizeObserver.observe(aiInputArea);
        }

        window.addEventListener('resize', () => adjustMessagesPadding());
        setTimeout(adjustMessagesPadding, 100);
    });
})();