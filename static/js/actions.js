// actions.js — полностью интернационализированная версия
(function() {
    if (window._actionsLoaded) return;
    window._actionsLoaded = true;

    // Helper for i18n
    function t(key, opts) { return i18next.t(key, opts); }

    // ========== Глобальные переменные ==========
    let pendingFile = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    // ========== Автоматическое расширение textarea ==========
    function autoResizeTextarea(textarea) {
        if (!textarea) return;
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 150);
        textarea.style.height = newHeight + 'px';
    }

    // ========== Вспомогательные функции ==========
    async function uploadEncryptedFile(file) {
        const { key, iv } = DarkCrypto.generateFileKeyAndIv();
        const fileData = await file.arrayBuffer();
        const encrypted = await DarkCrypto.encryptFile(new Uint8Array(fileData), key, iv);
        const blob = new Blob([encrypted], { type: 'application/octet-stream' });
        const formData = new FormData();
        formData.append('file', blob, 'encrypted.bin');
        const res = await fetch('/upload_encrypted', { method: 'POST', body: formData });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        return {
            url: data.file_url,
            key: DarkCrypto.arrayBufferToBase64(key),
            iv: DarkCrypto.arrayBufferToBase64(iv)
        };
    }

    function showFilePreview(file, type) {
    const oldPreview = document.getElementById('filePreview');
    if (oldPreview) oldPreview.remove();

    const previewContainer = document.createElement('div');
    previewContainer.id = 'filePreview';
    previewContainer.style.cssText = `
        display: flex; align-items: center; gap: 8px; padding: 8px 12px;
        margin: 0 16px 8px 16px; background: rgba(30,30,30,0.95);
        border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);
    `;

    let previewContent;
    let objectUrl = null;

    if (type.startsWith('image/')) {
        objectUrl = URL.createObjectURL(file);
        if (pendingFile) pendingFile._objectUrl = objectUrl;
        previewContent = `<img src="${objectUrl}" style="width: 32px; height: 32px; object-fit: cover; border-radius: 4px;">`;
    } else {
        previewContent = `<span>${type.startsWith('image') ? '🖼️' : '🎵'}</span>`;
    }

    previewContainer.innerHTML = `
        ${previewContent}
        <span style="flex:1; font-size:13px;">${Utils.escapeHtml(file.name)} (${(file.size/1024).toFixed(1)} KB)</span>
        <button id="cancelFileBtn" class="btn-icon-oval"><img src="/static/icons/Remove.png" width="20" height="20" alt="Remove" style="filter: invert(1);"></button>
    `;

    const form = document.querySelector('.chat-panel .input-area');
    if (form) {
        form.insertBefore(previewContainer, form.firstChild);
    }

    // ✅ Корректируем отступ ПОСЛЕ того, как блок реально добавился в DOM
    if (window.adjustMessagesPadding) window.adjustMessagesPadding();

    document.getElementById('cancelFileBtn')?.addEventListener('click', () => {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        pendingFile = null;
        previewContainer.remove();
        updateSendButtonVisibility();
        // ✅ После удаления блока тоже корректируем отступ
        if (window.adjustMessagesPadding) window.adjustMessagesPadding();
    });
}

    // ========== Запись аудио ==========
    async function startRecording() {
        if (isRecording) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                if (audioBlob.size > 2 * 1024 * 1024) {
                    window.NotificationManager?.showToast(t('audio_too_long', { max: 2 }), 'error');
                    stream.getTracks().forEach(t => t.stop());
                    isRecording = false;
                    return;
                }
                const file = new File([audioBlob], 'voice.webm', { type: 'audio/webm' });
                pendingFile = { file, type: 'audio/webm' };
                showFilePreview(file, 'audio/webm');
                updateSendButtonVisibility();
                stream.getTracks().forEach(t => t.stop());
                isRecording = false;
                document.getElementById('recordIndicator')?.remove();
            };
            mediaRecorder.start();
            isRecording = true;
            const indicator = document.createElement('div');
            indicator.id = 'recordIndicator';
            indicator.textContent = t('recording_indicator');
            indicator.style.cssText = 'position:fixed; bottom:80px; left:50%; transform:translateX(-50%); background:#f44336; color:#fff; padding:8px 16px; border-radius:20px; z-index:1000; cursor:pointer;';
            indicator.onclick = () => { if (mediaRecorder?.state === 'recording') mediaRecorder.stop(); };
            document.body.appendChild(indicator);
        } catch (err) {
            window.NotificationManager?.showToast(t('microphone_denied'), 'error');
        }
    }

    // ========== Выбор файла ==========
    function handleFileSelection(event, type) {
    const file = event.target.files[0];
    if (!file) return;

    // ── Проверка размера ──
    const maxSize = type === 'image' ? 10 * 1024 * 1024 : 2 * 1024 * 1024;
    if (file.size > maxSize) {
        window.NotificationManager?.showToast(
            t('file_too_large', { size: maxSize / 1024 / 1024 }),
            'error'
        );
        return;
    }

    // ── Проверка типа ──
    const allowedTypes = type === 'image'
        ? ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        : ['audio/webm', 'audio/mp4', 'audio/ogg'];
    if (!allowedTypes.includes(file.type)) {
        window.NotificationManager?.showToast(
            t('unsupported_file_type', { type: type }),
            'error'
        );
        return;
    }

    // ── Сжатие для изображений ──
    if (type === 'image' && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                // 1. Сжимаем изображение (maxWidth = 800-1200, quality = 0.7-0.90)
                const compressedDataUrl = await window.compressImage(
                    e.target.result,
                   1200,   // максимальная ширина/высота
                    0.85    // качество JPEG (0–1)
                );

                // 2. Преобразуем DataURL в Blob
                const res = await fetch(compressedDataUrl);
                const blob = await res.blob();

                // 3. Создаём новый File с правильным типом (JPEG)
                const compressedFile = new File(
                    [blob],
                    file.name.replace(/\.[^.]+$/, '.jpg'), // меняем расширение
                    { type: 'image/jpeg' }
                );

                // 4. Сохраняем сжатый файл как pendingFile
                pendingFile = { file: compressedFile, type: 'image/jpeg' };

                // 5. Показываем превью (уже сжатое)
                showFilePreview(compressedFile, 'image/jpeg');

                // 6. Обновляем кнопки
                updateSendButtonVisibility();
            } catch (err) {
                console.error('Compression error:', err);
                // Если сжатие не удалось – используем оригинал
                pendingFile = { file, type: file.type };
                showFilePreview(file, file.type);
                updateSendButtonVisibility();
                window.NotificationManager?.showToast(
                    t('compression_failed'),
                    'warning'
                );
            }
        };
        reader.readAsDataURL(file);
        // Важно: выходим, чтобы не выполнять код для несжатых файлов
        return;
    }

    // ── Для аудио и прочих файлов сжатие не применяем ──
    pendingFile = { file, type: file.type };
    showFilePreview(file, file.type);
    updateSendButtonVisibility();
    event.target.value = '';
}

    // ========== Функция управления видимостью кнопок ==========
    function updateSendButtonVisibility() {
        const sendBtn = document.getElementById('sendButton');
        const recordBtn = document.getElementById('recordAudioButton');
        if (!sendBtn || !recordBtn) return;

        const messageInput = document.getElementById('messageContent');
        const hasText = messageInput && messageInput.value.trim() !== '';
        const hasFile = pendingFile !== null;

        if (hasText || hasFile) {
            sendBtn.style.display = 'flex';
            recordBtn.style.display = 'none';
        } else {
            sendBtn.style.display = 'none';
            recordBtn.style.display = 'flex';
        }
    }

    // ========== Отправка сообщения ==========
    async function sendMessage() {
    if (State.currentChatAddress === 'ai_bot') return;
    if (window.isSending) return;
    const contentEl = document.getElementById('messageContent');
    let content = contentEl ? contentEl.value.trim() : '';
    if (!content && !pendingFile) {
        window.NotificationManager?.showToast(t('enter_message_or_attach'), 'warning');
        return;
    }

    window.isSending = true;
    const recipient = State.currentChatAddress;
    const isGroup = State.currentChatIsGroup;
    const groupId = isGroup && recipient.startsWith('group:') ? recipient.split(':')[1] : null;

    if (contentEl) { contentEl.value = ''; contentEl.style.height = 'auto'; }
    const fileToSend = pendingFile;
    pendingFile = null;
    if (fileToSend && fileToSend._objectUrl) {
        URL.revokeObjectURL(fileToSend._objectUrl);
    }
    const previewDiv = document.getElementById('filePreview');
    if (previewDiv) {
        const img = previewDiv.querySelector('img');
        if (img && img.src.startsWith('blob:')) {
            URL.revokeObjectURL(img.src);
        }
        previewDiv.remove();
    }
    document.getElementById('filePreview')?.remove();

    const tempId = 'temp-' + Date.now();
    const tempMsg = { id: tempId, sender: State.userAddress, recipient, content, timestamp: Date.now()/1000, is_mine: true, status: 'sent' };
    const container = document.getElementById('messagesContainer');
    if (container) {
        const emptyState = container.querySelector('.empty-state');
        if (emptyState) emptyState.remove();
        const tempElement = window.createMessageElement(tempMsg);
        container.appendChild(tempElement);
        tempElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    try {
        const keys = await window.ensureKeys();
        let fileAttachment = null;
        if (fileToSend) {
            const { url, key, iv } = await uploadEncryptedFile(fileToSend.file);
            fileAttachment = { url, key, iv, type: fileToSend.type };
        }

        let payload = {};
        if (isGroup && groupId) {
            const gRes = await fetch('/get_groups');
            const gData = await gRes.json();
            const freshGroup = gData.groups?.find(g => g.id === groupId);
            const members = freshGroup?.members || [];
            if (!members.length) throw new Error(t('group_members_not_loaded'));
            const encryptedMap = {};
            for (const addr of members) {
                const pubKeyB64 = await window.getPubKey(addr);
                const pubKeyBytes = DarkCrypto._fromBase64(pubKeyB64);
                const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, pubKeyBytes);
                let encryptedText = null;
                if (content) {
                    const { ciphertext, iv: textIv } = await DarkCrypto.encryptAES(shared, content);
                    encryptedText = { ciphertext: DarkCrypto._arrayBufferToBase64(ciphertext), iv: DarkCrypto._toBase64(textIv) };
                }
                let encFileKey = null, encFileIv = null;
                if (fileAttachment) {
                    const fileKeyBuffer = DarkCrypto.base64ToArrayBuffer(fileAttachment.key);
                    const fileIvBuffer = DarkCrypto.base64ToArrayBuffer(fileAttachment.iv);
                    const encKey = await DarkCrypto.encryptAES(shared, DarkCrypto.arrayBufferToBase64(new Uint8Array(fileKeyBuffer)));
                    const encIv = await DarkCrypto.encryptAES(shared, DarkCrypto.arrayBufferToBase64(new Uint8Array(fileIvBuffer)));
                    encFileKey = { ciphertext: DarkCrypto._arrayBufferToBase64(encKey.ciphertext), iv: DarkCrypto._toBase64(encKey.iv) };
                    encFileIv = { ciphertext: DarkCrypto._arrayBufferToBase64(encIv.ciphertext), iv: DarkCrypto._toBase64(encIv.iv) };
                }
                encryptedMap[addr] = {
                    text: encryptedText,
                    file_url: fileAttachment?.url,
                    file_key: encFileKey,
                    file_iv: encFileIv,
                    file_type: fileAttachment?.type,
                    sender_pubkey: DarkCrypto._toBase64(keys.compressedPubKey)
                };
                if (addr === State.userAddress) {
                    encryptedMap[addr].self_text = content ? { ciphertext: encryptedText.ciphertext, iv: encryptedText.iv } : null;
                    if (fileAttachment) {
                        encryptedMap[addr].self_file_key = fileAttachment.key;
                        encryptedMap[addr].self_file_iv = fileAttachment.iv;
                    }
                }
            }
            payload = { message_type: 'group', group_id: groupId, encrypted_map: encryptedMap };
        } else {
            const pubRes = await fetch(`/get_public_key/${recipient}`);
            if (!pubRes.ok) throw new Error(t('recipient_pubkey_not_found'));
            const pubData = await pubRes.json();
            const recipientPubKeyBytes = DarkCrypto._fromBase64(pubData.public_key);
            const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, recipientPubKeyBytes);
            let encryptedText = null;
            if (content) {
                const { ciphertext, iv } = await DarkCrypto.encryptAES(shared, content);
                encryptedText = { ciphertext: DarkCrypto._arrayBufferToBase64(ciphertext), iv: DarkCrypto._toBase64(iv) };
            }
            let encFileKey = null, encFileIv = null;
            if (fileAttachment) {
                const fileKeyBuffer = DarkCrypto.base64ToArrayBuffer(fileAttachment.key);
                const fileIvBuffer = DarkCrypto.base64ToArrayBuffer(fileAttachment.iv);
                const encKey = await DarkCrypto.encryptAES(shared, DarkCrypto.arrayBufferToBase64(new Uint8Array(fileKeyBuffer)));
                const encIv = await DarkCrypto.encryptAES(shared, DarkCrypto.arrayBufferToBase64(new Uint8Array(fileIvBuffer)));
                encFileKey = { ciphertext: DarkCrypto._arrayBufferToBase64(encKey.ciphertext), iv: DarkCrypto._toBase64(encKey.iv) };
                encFileIv = { ciphertext: DarkCrypto._arrayBufferToBase64(encIv.ciphertext), iv: DarkCrypto._toBase64(encIv.iv) };
            }
            const selfShared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, keys.compressedPubKey);
            let selfEncText = null, selfFileKey = null, selfFileIv = null;
            if (content) {
                const { ciphertext, iv } = await DarkCrypto.encryptAES(selfShared, content);
                selfEncText = { ciphertext: DarkCrypto._arrayBufferToBase64(ciphertext), iv: DarkCrypto._toBase64(iv) };
            }
            if (fileAttachment) {
                selfFileKey = fileAttachment.key;
                selfFileIv = fileAttachment.iv;
            }
            payload = {
                recipient: recipient,
                payload: {
                    text: encryptedText,
                    file_url: fileAttachment?.url,
                    file_key: encFileKey,
                    file_iv: encFileIv,
                    file_type: fileAttachment?.type,
                    sender_pubkey: DarkCrypto._toBase64(keys.compressedPubKey),
                    self_text: selfEncText,
                    self_file_key: selfFileKey,
                    self_file_iv: selfFileIv
                },
                message_type: 'direct'
            };
        }

        const res = await fetch('/send_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
            const sentMessage = {
                id: data.tx_id,
                sender: State.userAddress,
                recipient: recipient,
                content: content,
                timestamp: Date.now() / 1000,
                is_mine: true,
                status: 'sent',
                isDecrypted: true
            };

            // ✅ ДОБАВЛЯЕМ ДАННЫЕ ПРИКРЕПЛЁННОГО ФАЙЛА (если был)
            if (fileAttachment) {
                sentMessage.fileUrl = fileAttachment.url;
                sentMessage.fileKey = fileAttachment.key;
                sentMessage.fileIv = fileAttachment.iv;
                sentMessage.fileType = fileAttachment.type;
            }

            window.addMessageToCache(recipient, sentMessage, 'end');

            // Заменяем временный элемент реальным
            const tempElem = document.getElementById('msg-' + tempId);
            const realElem = window.createMessageElement(sentMessage);
            if (tempElem && tempElem.parentNode) {
                tempElem.parentNode.replaceChild(realElem, tempElem);
            } else {
                const container = document.getElementById('messagesContainer');
                if (container) container.appendChild(realElem);
            }

            // Обновляем превью в списке разговоров
            window.updateConversationPreview(recipient, content.slice(0, 40) || t('sent_preview'));

            // КОРРЕКТИРОВКА: принудительно обновляем отступ и прокручиваем вниз
            if (window.adjustMessagesPadding) window.adjustMessagesPadding();
            const msgContainer = document.getElementById('messagesContainer');
            if (msgContainer) {
                msgContainer.scrollTo({ top: msgContainer.scrollHeight, behavior: 'smooth' });
            }
        } else {
            document.getElementById('msg-' + tempId)?.remove();
            window.NotificationManager?.showToast(data.error || t('send_failed'), 'error');
            if (window.adjustMessagesPadding) window.adjustMessagesPadding();
        }
    } catch (err) {
        console.error(err);
        document.getElementById('msg-' + tempId)?.remove();
        window.NotificationManager?.showToast(err.message, 'error');
        if (window.adjustMessagesPadding) window.adjustMessagesPadding();
    } finally {
        window.isSending = false;
        const sendBtn = document.getElementById('sendButton');
        if (sendBtn) sendBtn.disabled = false;
        document.getElementById('messageContent')?.focus();
        updateSendButtonVisibility();
        setTimeout(() => {
            if (window.adjustMessagesPadding) window.adjustMessagesPadding();
        }, 50);
    }
}

    // ========== Модальные окна нового чата ==========
    function openNewChatModal() {
        window.modalOpen = true;
        document.getElementById('newChatModal')?.classList.remove('hidden');
        document.getElementById('newChatSelect').value = '';
        document.getElementById('newChatAddress').value = '';
        loadContactsForModal();

    }

    function closeNewChatModal() {
        window.modalOpen = false;
        document.getElementById('newChatModal')?.classList.add('hidden');

    }

    async function loadContactsForModal() {
        try {
            const res = await fetch('/get_contacts');
            const data = await res.json();
            if (res.ok && data.contacts) {
                State.allContacts = data.contacts;
                const select = document.getElementById('newChatSelect');
                if (select) {
                    select.innerHTML = `<option value="">${t('choose_contact')}</option>`;
                    data.contacts.forEach(c => {
                        const option = document.createElement('option');
                        option.value = c.address;
                        const name = c.name.length > 30 ? c.name.slice(0,27)+'…' : c.name;
                        option.textContent = Utils.escapeHtml(name) + ' (' + c.address.slice(0,10) + '…)';
                        select.appendChild(option);
                    });
                }
            }
        } catch (error) {
            console.error('Load contacts error:', error);
        }
    }

    async function startNewChat() {
        const select = document.getElementById('newChatSelect');
        const addressInput = document.getElementById('newChatAddress');
        const selected = select?.value.trim() || '';
        const entered = addressInput?.value.trim() || '';
        let address = '', name = '';
        if (selected) {
            address = selected;
            const contact = State.allContacts.find(c => c.address === selected);
            name = contact ? contact.name : selected.slice(0,10)+'…';
        } else if (entered) {
            const isValid = typeof Security !== 'undefined' ? Security.isValidAddress(entered) : /^[a-f0-9]{64}$/.test(entered);
            if (!isValid) {
                window.NotificationManager?.showToast(t('invalid_address_format'), 'error');
                return;
            }
            if (entered === State.userAddress) {
                window.NotificationManager?.showToast(t('cannot_chat_self'), 'warning');
                return;
            }
            address = entered;
            name = entered.slice(0,10)+'…';
        } else {
            window.NotificationManager?.showToast(t('select_or_enter_address'), 'warning');
            return;
        }
        closeNewChatModal();
        window.selectConversation(address, name, false);
    }

    // ========== Инициализация кнопок и авто-расширения ==========
    function initChatActions() {
        const msgInput = document.getElementById('messageContent');
        const aiInput = document.getElementById('aiMessageInput');
        if (msgInput) {
            msgInput.addEventListener('input', () => {
                autoResizeTextarea(msgInput);
                updateSendButtonVisibility();
            });
            msgInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }
        if (aiInput) {
            aiInput.addEventListener('input', () => autoResizeTextarea(aiInput));
        }

        const startChatBtn = document.getElementById('startNewChatBtn');
        if (startChatBtn) startChatBtn.onclick = startNewChat;

        const aiChatBtn = document.getElementById('aiChatBtn');
        if (aiChatBtn) {
            aiChatBtn.onclick = () => {
                window.selectConversation('ai_bot', t('ai_assistant'), false);
            };
        }

        const clearConvBtn = document.getElementById('clearConversationBtn');
        if (clearConvBtn) {
            clearConvBtn.onclick = async () => {
                if (!State.currentChatAddress) return;
                const confirmed = await window.showConfirmModal(t('clear_chat_title'), t('clear_chat_confirm'));
                if (confirmed) {
                    const res = await fetch('/clear_conversation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ chat_with: State.currentChatAddress })
                    });
                    if (res.ok) {
                        window.loadMessagesForConversation(State.currentChatAddress, false);
                        window.NotificationManager?.showToast(t('chat_cleared'), 'success');
                        const msgField = document.getElementById('messageContent');
                        if (msgField) msgField.value = '';
                        pendingFile = null;
                        document.getElementById('filePreview')?.remove();
                        updateSendButtonVisibility();
                    } else {
                        window.NotificationManager?.showToast(t('clear_failed'), 'error');
                    }
                }
            };
        }

        const addContactBtn = document.getElementById('addToContactsBtn');
        if (addContactBtn) {
            addContactBtn.onclick = async () => {
                const address = State.currentChatPartnerAddress;
                if (!address) return;

                // Check if already in contacts
                try {
                    const res = await fetch('/get_contacts');
                    const data = await res.json();
                    if (res.ok && data.contacts) {
                        const alreadyExists = data.contacts.some(c => c.address === address);
                        if (alreadyExists) {
                            window.NotificationManager?.showToast(t('contact_already_exists'), 'warning');
                            return;
                        }
                    }
                } catch (err) {
                    console.warn('Failed to check contacts', err);
                }

                const name = await window.showPromptModal(
                    t('add_contact_title'),
                    t('enter_contact_name'),
                    address.slice(0, 10) + '...'
                );
                if (!name) return;

                const res = await fetch('/add_contact_from_chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ contact_address: address, contact_name: name })
                });
                if (res.ok) {
                    window.NotificationManager?.showToast(t('contact_added'), 'success');
                    addContactBtn.disabled = true;
                    const refreshRes = await fetch('/get_contacts');
                    const refreshData = await refreshRes.json();
                    if (refreshRes.ok) State.allContacts = refreshData.contacts;
                } else {
                    const err = await res.json();
                    window.NotificationManager?.showToast(err.error || t('failed'), 'error');
                }
            };
        }

        const attachImageBtn = document.getElementById('attachImageButton');
        const imageInput = document.getElementById('imageInput');
        if (attachImageBtn && imageInput) {
            attachImageBtn.onclick = () => imageInput.click();
            imageInput.onchange = (e) => handleFileSelection(e, 'image');
        }
        const audioBtn = document.getElementById('attachAudioButton');
        const audioInput = document.getElementById('audioInput');
        if (audioBtn && audioInput) {
            audioBtn.onclick = () => audioInput.click();
            audioInput.onchange = (e) => handleFileSelection(e, 'audio');
        }
        const recordBtn = document.getElementById('recordAudioButton');
        if (recordBtn) recordBtn.onclick = startRecording;

        const sendBtn = document.getElementById('sendButton');
        if (sendBtn) sendBtn.onclick = sendMessage;

        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) newChatBtn.onclick = openNewChatModal;

        updateSendButtonVisibility();
    }

    document.addEventListener('click', async (e) => {
        const deleteBtn = e.target.closest('.delete-btn');
        if (deleteBtn && deleteBtn.dataset.id) {
            e.preventDefault();
            const msgId = deleteBtn.dataset.id;
            const confirmed = await window.showConfirmModal(t('delete_message_title'), t('delete_message_confirm'));
            if (confirmed) {
                const res = await fetch('/delete_message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message_id: parseInt(msgId) })
                });
                if (res.ok) {
                    const msgDiv = document.getElementById('msg-' + msgId);
                    if (msgDiv) {
                         msgDiv.querySelectorAll('[data-object-url]').forEach(el => {
                            URL.revokeObjectURL(el.dataset.objectUrl);
                         });
                         msgDiv.remove();
                    }
                    window.loadConversations();
                } else {
                    window.NotificationManager?.showToast(t('delete_failed'), 'error');
                }
            }
        }

        // Закрыть контекстное меню при клике вне его
        const contextMenu = document.querySelector('.message-context-menu.active');
        if (contextMenu && !e.target.closest('.message-context-menu')) {
            contextMenu.classList.remove('active');
        }
    });

    // ========== Контекстное меню сообщений ==========
    // Единый движок: правый клик (ПК) + долгое нажатие (тач) — context-menu.js.

    function getMessageText(messageEl) {
        const p = messageEl.querySelector('.content p');
        if (!p) return '';
        // Сохраняем переносы строк (<br>) при копировании
        const withBreaks = p.innerHTML.replace(/<br\s*\/?>/gi, '\n');
        const tmp = document.createElement('div');
        tmp.innerHTML = withBreaks;
        let text = tmp.textContent || '';
        // Убираем пометку "(изменено)" / "(edited)", добавляемую при редактировании
        text = text.replace(/\s*[（(]?(изменено|отредактировано|edited)[）)]?\s*$/i, '');
        return text.trim();
    }

    function findCachedMessage(msgId) {
        const chatId = State.currentChatAddress;
        if (!chatId || !window.getCachedMessages) return null;
        return window.getCachedMessages(chatId).find(m => String(m.id) === String(msgId)) || null;
    }

    function applyMessageEditLocally(msgId, newText) {
        const sel = `.message[data-message-id="${msgId}"], .message[data-id="${msgId}"]`;
        document.querySelectorAll(sel).forEach(el => {
            const p = el.querySelector('.content p');
            if (p) {
                p.textContent = newText;
                if (!el.querySelector('.msg-edited-tag')) {
                    p.insertAdjacentHTML('afterend', `<span class="msg-edited-tag">${t('edited')}</span>`);
                }
            }
        });
        const cached = findCachedMessage(msgId);
        if (cached) cached.content = newText;
    }

    async function doDeleteMessage(msgId) {
        const confirmed = await window.showConfirmModal(t('delete_message_title'), t('delete_message_confirm'));
        if (!confirmed) return;
        try {
            const res = await fetch('/delete_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message_id: parseInt(msgId) })
            });
            if (res.ok) {
                const msgDiv = document.getElementById('msg-' + msgId);
                if (msgDiv) {
                    msgDiv.querySelectorAll('[data-object-url]').forEach(el => URL.revokeObjectURL(el.dataset.objectUrl));
                    msgDiv.remove();
                }
                if (window.clearMessageCacheForId) window.clearMessageCacheForId(State.currentChatAddress, msgId);
                window.loadConversations();
                window.NotificationManager?.showToast(t('message_deleted'), 'success');
            } else {
                window.NotificationManager?.showToast(t('delete_failed'), 'error');
            }
        } catch (err) {
            console.error('Delete error:', err);
            window.NotificationManager?.showToast(t('delete_failed'), 'error');
        }
    }

    function fallbackCopy(text, done) {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.cssText = 'position:fixed;opacity:0;left:-9999px;';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done && done(); }
        catch (e) { window.NotificationManager?.showToast(t('copy_failed'), 'error'); }
        ta.remove();
    }

    // Единый набор SVG-иконок для контекстных меню (stroke через CSS).
    // ВАЖНО: иконки — только SVG, без PNG <img>, иначе в меню появляются «двойные» иконки.
    const CTX_ICONS = {
        copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
        reply: '<svg viewBox="0 0 24 24"><path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 5 5v1a5 5 0 0 1-5 5h-3"/></svg>',
        edit: '<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>',
        pin: '<svg viewBox="0 0 24 24"><path d="M12 17v5"/><path d="M9 10.8V4h6v6.8l2 2.2H7l2-2.2Z"/></svg>',
        unpin: '<svg viewBox="0 0 24 24"><path d="M12 17v5"/><path d="M9 10.8V4h6v6.8l2 2.2H7l2-2.2Z"/><path d="M4 4l16 16" stroke-width="1.6"/></svg>',
        trash: '<svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>',
        open: '<svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
        contact: '<svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        call: '<svg viewBox="0 0 24 24"><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 2 .7 2.9a2 2 0 0 1-.4 2.1L8.1 10a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.9.6 2.9.7a2 2 0 0 1 1.6 1.9z"/></svg>',
        info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
        hide: '<svg viewBox="0 0 24 24"><path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c7 0 10 8 10 8a18 18 0 0 1-2.16 3.19"/><path d="M6.61 6.61A18 18 0 0 0 2 12s3 8 10 8a9 9 0 0 0 5.39-1.61"/><path d="M14.12 14.12A3 3 0 1 1 9.88 9.88"/><path d="M2 2l20 20"/></svg>'
    };

    // --- Единое хранилище закреплённых сообщений (по одному на чат) ---
    window.getPinnedForChat = function (chatAddress) {
        if (!chatAddress) return null;
        try {
            const raw = localStorage.getItem('pinned_' + chatAddress);
            return raw ? JSON.parse(raw) : null;
        } catch (e) { return null; }
    };
    window.setPinnedForChat = function (chatAddress, data) {
        if (!chatAddress) return;
        const key = 'pinned_' + chatAddress;
        if (data) localStorage.setItem(key, JSON.stringify(data));
        else localStorage.removeItem(key);
        if (window.updatePinnedMessageBar) window.updatePinnedMessageBar(chatAddress);
    };

    function buildMessageMenuItems(messageEl) {
        const msgId = messageEl.dataset.messageId || messageEl.dataset.id;
        if (!msgId || String(msgId).startsWith('temp')) return null; // временные сообщения — без меню

        const isOwn = messageEl.classList.contains('sent') || messageEl.classList.contains('message-own');
        const messageContent = getMessageText(messageEl);
        let isPinned = false;
        try {
            const pinnedData = window.getPinnedForChat ? window.getPinnedForChat(State.currentChatAddress) : null;
            isPinned = !!(pinnedData && String(pinnedData.messageId) === String(msgId));
        } catch (e) {}

        return [
            {
                icon: CTX_ICONS.copy,
                label: t('copy'),
                onClick: () => {
                    const done = () => window.NotificationManager?.showToast(t('copied_to_clipboard'), 'success');
                    if (navigator.clipboard?.writeText) {
                        navigator.clipboard.writeText(messageContent).then(done).catch(() => fallbackCopy(messageContent, done));
                    } else {
                        fallbackCopy(messageContent, done);
                    }
                }
            },
            {
                icon: CTX_ICONS.reply,
                label: t('reply'),
                onClick: () => {
                    const senderName = messageEl.querySelector('.content strong')?.textContent || '';
                    const quoted = messageContent.split('\n').map(l => '> ' + l).join('\n');
                    const textarea = document.getElementById('messageContent');
                    if (textarea) {
                        textarea.value += (textarea.value ? '\n' : '') + (senderName ? senderName + ':\n' : '') + quoted + '\n';
                        textarea.focus();
                        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
                        if (window.autoResizeTextarea) window.autoResizeTextarea(textarea);
                        updateSendButtonVisibility();
                    }
                }
            },
            {
                icon: CTX_ICONS.edit,
                label: t('edit'),
                hidden: !isOwn,
                onClick: async () => {
                    const newText = await window.showPromptModal(t('edit'), t('edit_message_prompt'), messageContent);
                    if (newText === null || newText.trim() === '' || newText.trim() === messageContent.trim()) return;
                    try {
                        const res = await fetch('/edit_message', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message_id: parseInt(msgId), content: newText.trim() })
                        });
                        if (res.ok) {
                            applyMessageEditLocally(msgId, newText.trim());
                            window.NotificationManager?.showToast(t('message_edited'), 'success');
                        } else {
                            window.NotificationManager?.showToast(t('edit_failed'), 'error');
                        }
                    } catch (err) {
                        console.error('Edit error:', err);
                        window.NotificationManager?.showToast(t('edit_failed'), 'error');
                    }
                }
            },
            { separator: true },
            {
                icon: isPinned ? CTX_ICONS.unpin : CTX_ICONS.pin,
                label: isPinned ? t('unpin_message') : t('pin_message'),
                onClick: () => {
                    const chatAddress = State.currentChatAddress;
                    if (!chatAddress) {
                        window.NotificationManager?.showToast(t('open_chat_first'), 'warning');
                        return;
                    }
                    if (isPinned) {
                        // Открепление текущего сообщения
                        window.setPinnedForChat(chatAddress, null);
                        messageEl.classList.remove('pinned-highlight');
                        window.NotificationManager?.showToast(t('unpinned_message'), 'success');
                    } else {
                        // Закрепление (заменяет предыдущее закреплённое в этом чате)
                        window.setPinnedForChat(chatAddress, {
                            messageId: msgId,
                            content: messageContent.substring(0, 100),
                            timestamp: Date.now()
                        });
                        messageEl.classList.add('pinned-highlight');
                        setTimeout(() => messageEl.classList.remove('pinned-highlight'), 2000);
                        window.NotificationManager?.showToast(t('pinned_message'), 'success');
                    }
                }
            },
            {
                icon: CTX_ICONS.trash,
                label: t('delete'),
                danger: true,
                hidden: !isOwn,
                onClick: () => doDeleteMessage(msgId)
            }
        ];
    }

    // Правый клик по сообщению (делегирование — работает и для новых сообщений из WebSocket)
    document.addEventListener('contextmenu', (e) => {
        const messageEl = e.target.closest('.message');
        if (!messageEl || !window.ContextMenu) return;
        e.preventDefault();
        const items = buildMessageMenuItems(messageEl);
        if (!items) return;
        window.ContextMenu.open(items, { x: e.clientX, y: e.clientY }, { target: messageEl });
    }, { passive: false });

    // Долгое нажатие по сообщению (тач/стикус) — вешаем при создании элемента
    window.bindMessageHold = function (messageEl) {
        if (!messageEl || !window.ContextMenu || messageEl._ctxBound) return;
        messageEl._ctxBound = true;
        window.ContextMenu.bind(messageEl, () => buildMessageMenuItems(messageEl));
    };

    // Очистка кэша сообщений при удалении
    window.clearMessageCacheForId = function (chatId, msgId) {
        if (!chatId || !window.messagesCache) return;
        const arr = window.messagesCache.get(chatId);
        if (arr) {
            const idx = arr.findIndex(m => String(m.id) === String(msgId));
            if (idx >= 0) arr.splice(idx, 1);
            const idSet = window._messageIdSets?.get(chatId);
            if (idSet) { idSet.delete(parseInt(msgId)); idSet.delete(msgId); }
        }
    };

    // Unpin button handler
    document.addEventListener('DOMContentLoaded', function() {
        const unpinBtn = document.getElementById('unpinMessageBtn');
        if (unpinBtn) {
            unpinBtn.addEventListener('click', () => {
                const chatAddress = State.currentChatAddress;
                if (chatAddress) {
                    window.unpinMessage(chatAddress);
                }
            });
        }
    });

    document.addEventListener('DOMContentLoaded', initChatActions);
    window.sendMessage = sendMessage;
    window.handleFileSelection = handleFileSelection;
    window.openNewChatModal = openNewChatModal;
    window.closeNewChatModal = closeNewChatModal;
    window.startNewChat = startNewChat;
    window.autoResizeTextarea = autoResizeTextarea;
    window.updateSendButtonVisibility = updateSendButtonVisibility;
    window.unpinMessage = unpinMessage;
    
    // Theme toggle function
    window.toggleTheme = function() {
        const body = document.body;
        const isLight = body.classList.toggle('light-theme');
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
        
        // Update icon visibility
        const sunIcon = document.querySelector('.sun-icon');
        const moonIcon = document.querySelector('.moon-icon');
        if (sunIcon && moonIcon) {
            sunIcon.classList.toggle('hidden', isLight);
            moonIcon.classList.toggle('hidden', !isLight);
        }
    };
    
    // Add event listener for theme toggle button
    document.addEventListener('DOMContentLoaded', function() {
        const themeBtn = document.getElementById('themeToggleBtn');
        if (themeBtn) {
            themeBtn.addEventListener('click', window.toggleTheme);
        }
    });
    
    // Initialize theme from localStorage
    (function initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            const sunIcon = document.querySelector('.sun-icon');
            const moonIcon = document.querySelector('.moon-icon');
            if (sunIcon && moonIcon) {
                sunIcon.classList.add('hidden');
                moonIcon.classList.remove('hidden');
            }
        }
    })();
    
    // Search chats function
    window.filterChats = function(query) {
        const list = document.getElementById('conversationsList');
        if (!list) return;
        const items = list.querySelectorAll('.conversation-item');
        const lowerQuery = query.toLowerCase();
        
        items.forEach(item => {
            const name = item.querySelector('.name')?.textContent?.toLowerCase() || '';
            const meta = item.querySelector('.meta')?.textContent?.toLowerCase() || '';
            if (name.includes(lowerQuery) || meta.includes(lowerQuery)) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    };
    
    // Pinned message bar (single source of truth: localStorage pinned_<chat>)
    window.updatePinnedMessageBar = function(chatAddress) {
        const bar = document.getElementById('pinnedMessagesBar');
        const preview = document.getElementById('pinnedMessagePreview');
        if (!bar || !preview) return;

        let data = null;
        try {
            const raw = localStorage.getItem('pinned_' + chatAddress);
            data = raw ? JSON.parse(raw) : null;
        } catch (e) { data = null; }

        if (data && data.messageId) {
            preview.textContent = data.content || '';
            bar.classList.add('active');
            bar.dataset.pinnedMessageId = data.messageId;

            // Click on the bar scrolls to the pinned message
            preview.onclick = () => {
                const msgEl = document.getElementById('msg-' + data.messageId);
                if (msgEl) {
                    msgEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    msgEl.classList.add('pinned-highlight');
                    setTimeout(() => msgEl.classList.remove('pinned-highlight'), 2000);
                }
            };
        } else {
            bar.classList.remove('active');
            delete bar.dataset.pinnedMessageId;
            preview.onclick = null;
        }
    };

    window.unpinMessage = function(chatAddress) {
        localStorage.removeItem('pinned_' + chatAddress);
        window.updatePinnedMessageBar(chatAddress);
    };
})();