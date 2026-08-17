// core.js — ядро мессенджера с поддержкой зашифрованного localStorage и кеша сообщений
// Исправлено: сохранение статуса сообщений, разделители дат (вынесены в ui.js)
(function() {
    if (window._coreLoaded) return;
    window._coreLoaded = true;

    // Helper for safe i18n (fallback to English if i18next not ready)
    function t(key, opts) {
        if (typeof i18next !== 'undefined' && i18next.t) {
            return i18next.t(key, opts);
        }
        const fallbacks = {
            'unlock_wallet_title': 'Unlock wallet',
            'unlock_wallet_desc': 'Your encrypted wallet was found. Enter password to unlock.',
            'password': 'Password',
            'log_out': 'Log out',
            'unlock': 'Unlock',
            'please_enter_password': 'Please enter password',
            'wrong_password': 'Wrong password',
            'public_key_not_found': 'Public key not found for {address}',
            'public_key_mismatch': '⚠️ Public key mismatch for {address} — possible MITM!',
            'no_mnemonic_available': 'No mnemonic available',
            'mnemonic_not_found': 'Mnemonic not found',
            'no_access': '🔒 No access',
            'no_sender_pubkey': '🔒 No sender pubkey',
            'decrypt_error': '🔒 Decrypt error',
            'new_message_preview': '💬 New message',
            'read_status': '✓✓ Read',
            'delivered_status': '✓✓ Delivered',
            'sent_status': '✓ Sent',
            'online': 'Online',
            'offline': 'Offline'
        };
        let result = fallbacks[key];
        if (result && opts) {
            if (opts.address) result = result.replace('{address}', opts.address);
        }
        return result || key;
    }

    // ========== Глобальные утилиты ==========
    window.Utils = window.Utils || {};
    Utils.escapeHtml = function(str) {
        if (!str) return '';
        return str.replace(/[&<>]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]))
                  .replace(/['"]/g, m => ({ "'": '&#39;', '"': '&quot;' }[m]));
    };
    Utils.formatTimestamp = function(ts) {
        if (!ts) return '';
        const date = new Date(ts * 1000);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };
    Utils.copyToClipboard = function(text, onSuccess, onError) {
        navigator.clipboard.writeText(text).then(onSuccess).catch(onError);
    };

    // ========== Глобальное состояние ==========
    let initialAddress = '';
    const metaAddress = document.querySelector('meta[name="user-address"]')?.content;
    if (metaAddress && metaAddress !== 'None' && metaAddress !== '') {
        initialAddress = metaAddress;
    }

    window.State = {
        currentChatAddress: '',
        currentChatIsGroup: false,
        currentGroupMembers: null,
        currentChatPartnerAddress: '',
        userAddress: initialAddress || (document.getElementById('app')?.dataset.userAddress) || '',
        lastMessageTimestamp: 0,
        allContacts: [],
        lastKnownMessageId: 0,
        pendingImageData: null,
        topObserver: null,
        _restoringMnemonic: false
    };
    window.isSending = false;
    window.userKeys = null;
    window.pubKeyCache = new Map();

    // ========== Кеш сообщений (в памяти, на время сессии) ==========
    window.messagesCache = new Map();
    // ИСПРАВЛЕНИЕ: Отдельный Set для O(1) проверки дубликатов вместо O(n) поиска по массиву
    window._messageIdSets = new Map();

    // Флаг, чтобы не обрабатывать звонок дважды
    window._pendingCallHandled = false;

    window.handlePendingCall = function() {
    if (window._pendingCallHandled) return;

    let callId = sessionStorage.getItem('pending_call_id');
    if (!callId) {
        const urlParams = new URLSearchParams(window.location.search);
        callId = urlParams.get('call_id');
    }
    if (!callId) return;

    sessionStorage.setItem('pending_call_id', callId);

    // Удаляем параметр из URL чтобы не было повторной обработки
    const newUrl = window.location.pathname;
    window.history.replaceState({}, document.title, newUrl);

    const trySendGetCall = () => {
        if (window.wsClient && window.wsClient.isConnected === true) {
            console.log('[App] Sending get_call for', callId);
            window.wsClient.send({ type: 'get_call', call_id: callId });
            window._pendingCallHandled = true;
            sessionStorage.removeItem('pending_call_id');
            return true;
        }
        return false;
    };

    // Пытаемся отправить сразу
    if (trySendGetCall()) return;

    // ✅ ИСПРАВЛЕНИЕ: Если WS не готов — ждём с повторными попытками
    // Это критично для звонков — приложение только что открылось из push-уведомления
    let retries = 0;
    const maxRetries = 40; // ИСПРАВЛЕНИЕ: 20 секунд максимум
    const retryInterval = setInterval(() => {
    retries++;
    if (trySendGetCall() || retries >= maxRetries) {
        clearInterval(retryInterval);
        if (retries >= maxRetries) {
            console.warn('[App] Failed to send get_call — WS not ready after 20s');
        }
    }
    }, 500);
};
    window.addMessageToCache = function(chatId, message, position = 'end') {
        if (!chatId || !message || !message.id) return;
        let idSet = window._messageIdSets.get(chatId);
        let messages = window.messagesCache.get(chatId);
        if (!messages) {
            messages = [];
            window.messagesCache.set(chatId, messages);
            idSet = new Set();
            window._messageIdSets.set(chatId, idSet);
        }
        if (idSet.has(message.id)) return; // O(1) вместо messages.some()
        idSet.add(message.id);

        if (position === 'start') {
            messages.unshift(message);
        } else {
            messages.push(message);
        }
        messages.sort((a, b) => a.id - b.id);
    };

    window.addMessagesToCache = function(chatId, newMessages, position = 'end') {
        if (!chatId || !newMessages?.length) return;
        let idSet = window._messageIdSets.get(chatId);
        let messages = window.messagesCache.get(chatId);
        if (!messages) {
            messages = [];
            window.messagesCache.set(chatId, messages);
            idSet = new Set();
            window._messageIdSets.set(chatId, idSet);
        }
        if (position === 'start') {
            messages.unshift(...newMessages);
        } else {
            messages.push(...newMessages);
        }
        const unique = new Map();
        for (const msg of messages) unique.set(msg.id, msg);
        const sorted = Array.from(unique.values()).sort((a, b) => a.id - b.id);
        window.messagesCache.set(chatId, sorted);
        // Обновляем Set ID
        window._messageIdSets.set(chatId, new Set(sorted.map(m => m.id)));
    };

    window.getCachedMessages = function(chatId) {
        return window.messagesCache.get(chatId) || [];
    };

    window.clearMessageCache = function(chatId) {
        if (chatId) {
            window.messagesCache.delete(chatId);
            window._messageIdSets.delete(chatId);
        } else {
            window.messagesCache.clear();
            window._messageIdSets.clear();
        }
    };

    // ========== Управление зашифрованной мнемоникой ==========
    // ИСПРАВЛЕНИЕ: Глобальная очередь ожидания вместо polling setInterval
    let _mnemonicResolveQueue = [];

    async function restoreMnemonic() {
        if (sessionStorage.getItem('mnemonic')) return true;
        const publicPaths = ['/login', '/', '/index', '/create_wallet'];
        const currentPath = window.location.pathname;
        if (publicPaths.some(p => currentPath === p || currentPath.startsWith(p + '?'))) {
            console.debug('Public page, skipping mnemonic restore');
            return false;
        }
        if (State._restoringMnemonic) {
            // Возвращаем Promise, который разрешится, когда юзер введет пароль
            return new Promise(resolve => {
                _mnemonicResolveQueue.push(resolve);
            });
        }
        State._restoringMnemonic = true;
        const encrypted = localStorage.getItem('encrypted_mnemonic');
        if (!encrypted) {
            window.location.href = '/login';
            State._restoringMnemonic = false;
            return false;
        }
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'modal-overlay';
            modal.style.zIndex = '10000';
            modal.innerHTML = `
                <div class="modal" style="max-width:400px;">
                    <div class="modal-header"><h3 data-i18n="unlock_wallet_title">Unlock wallet</h3></div>
                    <div class="modal-body">
                        <p data-i18n="unlock_wallet_desc">Your encrypted wallet was found. Enter password to unlock.</p>
                        <input type="password" id="unlockPassword" class="input" placeholder="Password" style="width:100%;" data-i18n-placeholder="password">
                        <div id="unlockError" class="text-error" style="color:var(--danger);margin-top:8px;display:none;"></div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-ghost" id="cancelUnlock" data-i18n="log_out">Log out</button>
                        <button class="btn btn-primary" id="confirmUnlock" data-i18n="unlock">Unlock</button>
                    </div>
                </div>`;

            document.body.appendChild(modal);
            if (window.localizePage) window.localizePage();
            const passwordInput = modal.querySelector('#unlockPassword');
            const errorDiv = modal.querySelector('#unlockError');
            const confirmBtn = modal.querySelector('#confirmUnlock');
            const cancelBtn = modal.querySelector('#cancelUnlock');

            const attemptUnlock = async () => {
                const pwd = passwordInput.value;
                if (!pwd) {
                    errorDiv.textContent = t('please_enter_password');
                    errorDiv.style.display = 'block';
                    return;
                }
                confirmBtn.disabled = true;
                confirmBtn.textContent = '...';

                try {
                    const mnemonic = await window.StorageEncryption.decryptMnemonic(encrypted, pwd);
                    if (mnemonic) {
                        sessionStorage.setItem('mnemonic', mnemonic);
                        modal.remove();
                        State._restoringMnemonic = false;

                        // ИСПРАВЛЕНИЕ: Разрешаем все обещания, кто ждал мнемонику
                        _mnemonicResolveQueue.forEach(r => r(true));
                        _mnemonicResolveQueue = [];

                        // 1. Инициализируем Push (если разрешение уже дано)
                        if (window.initPushNotifications) {
                            window.initPushNotifications().catch(e => console.warn('Push init after unlock:', e));
                        }

                        // 2. Инициализируем WebSocket сразу после получения ключа
                        if (typeof window.initWebSocket === 'function') {
                            console.log('[App] Initializing WebSocket after successful unlock...');
                            await window.initWebSocket();
                        }

                        resolve(true);
                    } else {
                        errorDiv.textContent = t('wrong_password');
                        errorDiv.style.display = 'block';
                        confirmBtn.disabled = false;
                        confirmBtn.textContent = t('unlock');
                    }
                } catch (err) {
                    console.error('Unlock error:', err);
                    errorDiv.textContent = 'Error unlocking wallet';
                    errorDiv.style.display = 'block';
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = t('unlock');
                }
            };

            const clearAndLogout = () => {
                localStorage.removeItem('encrypted_mnemonic');
                sessionStorage.removeItem('mnemonic');
                window.location.href = '/login';
                State._restoringMnemonic = false;

                // Отклоняем ожидания при выходе
                _mnemonicResolveQueue.forEach(r => r(false));
                _mnemonicResolveQueue = [];

                resolve(false);
            };
            confirmBtn.onclick = attemptUnlock;
            cancelBtn.onclick = clearAndLogout;
            passwordInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') attemptUnlock(); });
        });
    }

    // ========== Криптографические помощники ==========
    async function getPubKey(address) {
        if (pubKeyCache.has(address)) return pubKeyCache.get(address);
        const res = await fetch(`/get_public_key/${address}`);
        if (!res.ok) throw new Error(t('public_key_not_found', { address }));
        const data = await res.json();
        const pubKeyBytes = DarkCrypto._fromBase64(data.public_key);
        const hashBuf = await crypto.subtle.digest('SHA-256', pubKeyBytes);
        const computedAddress = Array.from(new Uint8Array(hashBuf))
            .map(b => b.toString(16).padStart(2, '0')).join('');
        if (computedAddress !== address) {
            throw new Error(t('public_key_mismatch', { address }));
        }
        pubKeyCache.set(address, data.public_key);
        return data.public_key;
    }

    async function compressImage(dataUrl, maxWidth = 800, quality = 0.7) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let { width, height } = img;
                if (width > maxWidth || height > maxWidth) {
                    if (width > height) {
                        height = height * (maxWidth / width);
                        width = maxWidth;
                    } else {
                        width = width * (maxWidth / height);
                        height = maxWidth;
                    }
                }
                canvas.width = width;
                canvas.height = height;
                canvas.getContext('2d').drawImage(img, 0, 0, width, height);
                resolve(canvas.toDataURL('image/jpeg', quality));
            };
            img.onerror = reject;
            img.src = dataUrl;
        });
    }

    async function ensureKeys() {
        if (window.userKeys) return window.userKeys;
        if (!sessionStorage.getItem('mnemonic')) {
            const restored = await restoreMnemonic();
            if (!restored) throw new Error(t('no_mnemonic_available'));
        }
        const mnemonic = sessionStorage.getItem('mnemonic');
        if (!mnemonic) throw new Error(t('mnemonic_not_found'));
        window.userKeys = await DarkCrypto.deriveKeyPair(mnemonic);
        return window.userKeys;
    }

    // ========== WebSocket клиент ==========
    let wsClient = null;
    let wsReconnectTimer = null;

    async function initWebSocket() {
        if (!sessionStorage.getItem('mnemonic')) {
            const restored = await restoreMnemonic();
            if (!restored) return;
        }
        if (window.wsClient) window.wsClient.disconnect();
        let address = State.userAddress;
        if (!address) {
            const meta = document.querySelector('meta[name="user-address"]')?.content;
            if (meta && meta !== 'None') address = meta;
        }
        if (!address) {
            console.warn('No user address, WebSocket not initialized');
            return;
        }
        try {
            const keys = await ensureKeys();
            const nonce = crypto.randomUUID();
            const signatureArray = await DarkCrypto.signData(keys.signPrivateKey, nonce);
            const signatureHex = Array.from(new Uint8Array(signatureArray)).map(b => b.toString(16).padStart(2, '0')).join('');
            window.wsClient = new window.WebSocketClient({
                onMessage: window.handleWebSocketMessage,
                onConnect: () => {
                    console.log('✅ WebSocket connected');
                    if (window.loadConversations) window.loadConversations();
                    window.handlePendingCall();
                },
                onDisconnect: () => console.warn('⚠️ WebSocket disconnected'),
                onError: (err) => console.error('WebSocket error:', err)
            });
            window.wsClient.setAuth(address, signatureHex, nonce);
            window.wsClient.connect();
        } catch (err) { console.error('Failed to init WebSocket:', err); }
    }

    // ========== Обработка входящих сообщений (WebSocket + расшифровка файлов) ==========
    async function processMessageDecryption(msg) {
        if (!msg.content) return msg;
        let content = msg.content;
        let image = msg.image;
        let fileUrl = null, fileKey = null, fileIv = null, fileType = null;
        const originalStatus = msg.status || (msg.is_mine ? 'sent' : null);

        try {
            const parsed = JSON.parse(content);
            const keys = await ensureKeys();
            const arraysEqual = (a, b) => {
                if (!a || !b || a.length !== b.length) return false;
                for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
                return true;
            };

            // ---------- ГРУППОВОЙ ЧАТ ----------
            if (parsed.encrypted_map) {
                const myAddr = State.userAddress;
                const myEnc = parsed.encrypted_map[myAddr];
                if (!myEnc) return { ...msg, content: t('no_access'), status: originalStatus };
                const senderPubKeyBytes = DarkCrypto._fromBase64(myEnc.sender_pubkey);
                const isMine = arraysEqual(senderPubKeyBytes, keys.compressedPubKey);
                if (isMine && myEnc.self_text) {
                    const selfShared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, keys.compressedPubKey);
                    const ciphertext = DarkCrypto._base64ToArrayBuffer(myEnc.self_text.ciphertext);
                    const iv = DarkCrypto._fromBase64(myEnc.self_text.iv);
                    content = await DarkCrypto.decryptAES(selfShared, ciphertext, iv);
                } else if (myEnc.text) {
                    const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, senderPubKeyBytes);
                    const ciphertext = DarkCrypto._base64ToArrayBuffer(myEnc.text.ciphertext);
                    const iv = DarkCrypto._fromBase64(myEnc.text.iv);
                    content = await DarkCrypto.decryptAES(shared, ciphertext, iv);
                } else {
                    content = '';
                }
                if (myEnc.file_url) {
                    fileUrl = myEnc.file_url;
                    fileType = myEnc.file_type;
                    if (isMine && myEnc.self_file_key && myEnc.self_file_iv) {
                        fileKey = myEnc.self_file_key;
                        fileIv = myEnc.self_file_iv;
                    } else if (myEnc.file_key && myEnc.file_iv) {
                        const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, senderPubKeyBytes);
                        const keyCipher = DarkCrypto._base64ToArrayBuffer(myEnc.file_key.ciphertext);
                        const keyIv = DarkCrypto._fromBase64(myEnc.file_key.iv);
                        const decKey = await DarkCrypto.decryptAES(shared, keyCipher, keyIv);
                        const ivCipher = DarkCrypto._base64ToArrayBuffer(myEnc.file_iv.ciphertext);
                        const ivIv = DarkCrypto._fromBase64(myEnc.file_iv.iv);
                        const decIv = await DarkCrypto.decryptAES(shared, ivCipher, ivIv);
                        fileKey = decKey;
                        fileIv = decIv;
                    }
                }
                const chatId = msg.recipient;
                return { ...msg, content, image, fileUrl, fileKey, fileIv, fileType, is_mine: isMine, chatId, isGroup: true, isDecrypted: true, status: originalStatus };
            }

            // ---------- ЛИЧНЫЙ ЧАТ ----------
            const senderPubKeyB64 = parsed.sender_pubkey || (parsed.myPubKey ? parsed.myPubKey : null);
            if (!senderPubKeyB64) {
                if (parsed.file_url) {
                    fileUrl = parsed.file_url;
                    fileType = parsed.file_type;
                    if (parsed.self_file_key && parsed.self_file_iv) {
                        fileKey = parsed.self_file_key;
                        fileIv = parsed.self_file_iv;
                    }
                }
                const chatId = msg.sender === State.userAddress ? msg.recipient : msg.sender;
                return { ...msg, content: t('no_sender_pubkey'), image: null, fileUrl, fileKey, fileIv, fileType, is_mine: false, chatId, isDecrypted: false, status: originalStatus };
            }

            const senderPubKeyBytes = DarkCrypto._fromBase64(senderPubKeyB64);
            const isMine = arraysEqual(senderPubKeyBytes, keys.compressedPubKey);

            let decryptedText = '';
            if (parsed.text && parsed.text.ciphertext && parsed.text.iv) {
                if (isMine && parsed.self_text && parsed.self_text.ciphertext) {
                    const selfShared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, keys.compressedPubKey);
                    const ciphertext = DarkCrypto._base64ToArrayBuffer(parsed.self_text.ciphertext);
                    const iv = DarkCrypto._fromBase64(parsed.self_text.iv);
                    decryptedText = await DarkCrypto.decryptAES(selfShared, ciphertext, iv);
                } else {
                    const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, senderPubKeyBytes);
                    const ciphertext = DarkCrypto._base64ToArrayBuffer(parsed.text.ciphertext);
                    const iv = DarkCrypto._fromBase64(parsed.text.iv);
                    decryptedText = await DarkCrypto.decryptAES(shared, ciphertext, iv);
                }
                content = decryptedText;
            } else if (parsed.ciphertext && parsed.iv) {
                if (isMine && parsed.self_text && parsed.self_text.ciphertext) {
                    const selfShared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, keys.compressedPubKey);
                    const ciphertext = DarkCrypto._base64ToArrayBuffer(parsed.self_text.ciphertext);
                    const iv = DarkCrypto._fromBase64(parsed.self_text.iv);
                    decryptedText = await DarkCrypto.decryptAES(selfShared, ciphertext, iv);
                } else {
                    const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, senderPubKeyBytes);
                    const ciphertext = DarkCrypto._base64ToArrayBuffer(parsed.ciphertext);
                    const iv = DarkCrypto._fromBase64(parsed.iv);
                    decryptedText = await DarkCrypto.decryptAES(shared, ciphertext, iv);
                }
                content = decryptedText;
            } else {
                content = '';
            }

            if (parsed.file_url) {
                fileUrl = parsed.file_url;
                fileType = parsed.file_type;
                if (isMine && parsed.self_file_key && parsed.self_file_iv) {
                    fileKey = parsed.self_file_key;
                    fileIv = parsed.self_file_iv;
                } else if (parsed.file_key && parsed.file_iv) {
                    const shared = await DarkCrypto.getSharedSecret(keys.ecdhPrivateKey, senderPubKeyBytes);
                    const keyCipher = DarkCrypto._base64ToArrayBuffer(parsed.file_key.ciphertext);
                    const keyIv = DarkCrypto._fromBase64(parsed.file_key.iv);
                    const decKey = await DarkCrypto.decryptAES(shared, keyCipher, keyIv);
                    const ivCipher = DarkCrypto._base64ToArrayBuffer(parsed.file_iv.ciphertext);
                    const ivIv = DarkCrypto._fromBase64(parsed.file_iv.iv);
                    const decIv = await DarkCrypto.decryptAES(shared, ivCipher, ivIv);
                    fileKey = decKey;
                    fileIv = decIv;
                }
            }

            const chatId = msg.sender === State.userAddress ? msg.recipient : msg.sender;
            return { ...msg, content, image: null, fileUrl, fileKey, fileIv, fileType, is_mine: isMine, chatId, isDecrypted: true, status: originalStatus };
        } catch (e) {
            console.error('Decryption error', msg.id, e);
            const chatId = msg.sender === State.userAddress ? msg.recipient : msg.sender;
            return { ...msg, content: t('decrypt_error'), image: null, chatId, isDecrypted: false, status: originalStatus };
        }
    }

    async function handleWebSocketMessage(data) {
        if (data.error) {
            console.error('WS error:', data.error);
            return;
        }

        // 1. Обработка "call_not_found"
        if (data.type === 'call_not_found') {
            console.warn('[WS] Call not found:', data.call_id);
            sessionStorage.removeItem('pending_call_id');
            return;
        }

        // 2. Обработка сигналов звонка
        if (
            data.type === 'incoming_call' ||
            data.type === 'call_answer' ||
            data.type === 'call_ice' ||
            data.type === 'call_hangup' ||
            data.type === 'call_reject'
        ) {
            const tryCall = () => {
                if (window.handleCallSignal) {
                    window.handleCallSignal(data);
                } else {
                    console.warn('[WS] handleCallSignal not ready, retrying...');
                    setTimeout(tryCall, 100);
                }
            };
            tryCall();
            return;
        }

        // 3. Обычные сообщения
        if (data.type === 'message') {
            const decrypted = await processMessageDecryption(data);
            const chatId = decrypted.chatId;

            if (decrypted?.id) {
                window.addMessageToCache(chatId, decrypted, 'end');
            }

            if (!decrypted.is_mine) {
                fetch(`/message/${decrypted.id}/delivered`, { method: 'POST' }).catch(() => {});
            }

            if (window.onNewMessageReceived) {
                window.onNewMessageReceived(decrypted);
            }
            return;
        }

        // 4. Статус online/offline
        if (data.type === 'status_update') {
            if (data.address && data.status) {
                if (window.updateConversationStatus) {
                    window.updateConversationStatus(data.address, data.status);
                }
            }
            return;
        }

        // 5. Анти-дублирование push vs websocket
        if (data._from_push === true) {
            return;
        }

        // 6. Статус сообщения (delivered / read)
        if (data.type === 'message_status') {
            const msgDiv = document.querySelector(`.message-own[data-id="${data.message_id}"]`);
            if (msgDiv && msgDiv.dataset.status !== data.status) {
                msgDiv.dataset.status = data.status;
                if (window.updateStatusIcon) window.updateStatusIcon(msgDiv, data.status);
                const chatId = State.currentChatAddress;
                const cached = window.getCachedMessages ? window.getCachedMessages(chatId) : [];
                const cachedMsg = cached.find(m => m.id === data.message_id);
                if (cachedMsg) cachedMsg.status = data.status;
            }
            return;
        }

        // ====== НОВЫЙ БЛОК: Уведомления о транзакциях ======
        if (data.type === 'new_transaction') {
    const tx = data.transaction;
    // Проверяем, что транзакция касается нас
    if (tx && (tx.recipient === State.userAddress || tx.sender === State.userAddress)) {
        const amount = (tx.amount / 1_000_000).toFixed(6);
        const isIncome = tx.recipient === State.userAddress;
        const sign = isIncome ? '+' : '-';
        const label = isIncome ? t('received') : t('sent');
        window.NotificationManager?.showToast(
    `${label} ${sign}${amount} BlockCoin`,
    isIncome ? 'success' : 'info',
    8000  // ← 8 секунд
);
        // Обновляем баланс и историю
        if (window.refreshBalance) window.refreshBalance();
        if (window.loadTx) window.loadTx();
    }
    return; // не продолжаем обработку
}

        // 7. Старые сообщения (без chatId, без типа) – обратная совместимость
        if (!data.chatId && data.sender && data.recipient) {
            data.chatId = (data.sender === State.userAddress) ? data.recipient : data.sender;
        }
        if (!data.chatId) return;
        if (document.getElementById('msg-' + data.id)) return;

        const decrypted = await processMessageDecryption(data);
        const chatId = decrypted.chatId;

        if (decrypted && decrypted.id) window.addMessageToCache(chatId, decrypted, 'end');

        if (!decrypted.is_mine) {
            fetch(`/message/${decrypted.id}/delivered`, { method: 'POST' }).catch(e => console.warn(e));
        }

        const isCurrent = State.currentChatAddress === chatId ||
            (!decrypted.isGroup && (
                decrypted.sender === State.currentChatAddress ||
                decrypted.recipient === State.currentChatAddress
            ));

        if (!isCurrent) {
            const existingItem = document.querySelector(`.conversation-item[data-address="${chatId}"]`);
            if (!existingItem) {
                if (window.loadConversations) window.loadConversations();
            } else {
                if (window.updateConversationPreview) {
                    window.updateConversationPreview(chatId, decrypted.preview || t('new_message_preview'));
                }
                if (window.moveConversationToTop) {
                    window.moveConversationToTop(chatId);
                } else {
                    if (window.loadConversations) window.loadConversations();
                }
            }
        }

        if (isCurrent && window.onNewMessageReceived) window.onNewMessageReceived(decrypted);

        if (window.NotificationManager) {
            const previewText = (decrypted.content && !decrypted.content.startsWith('{'))
                ? decrypted.content.slice(0, 80)
                : (decrypted.fileUrl ? '📎 File' : '💬 New message');
            window.NotificationManager.handleIncomingMessage?.({
                sender: decrypted.sender,
                sender_name: decrypted.sender_name,
                chatId: chatId,
                isGroup: !!(decrypted.isGroup || decrypted.group_id),
                content: previewText,
                preview: previewText,
                timestamp: decrypted.timestamp * 1000,
                messageId: decrypted.id
            });
        }
    }

    // ========== Heartbeat ==========
    let heartbeatInterval = null;
    function startHeartbeat() {
        if (heartbeatInterval) clearInterval(heartbeatInterval);
        heartbeatInterval = setInterval(async () => {
            if (!State.userAddress) return;
            try {
                await fetch('/heartbeat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ current_chat: State.currentChatAddress || '' })
                });
            } catch(e) { console.debug('Heartbeat failed', e); }
        }, 30000);
    }
    function stopHeartbeat() { if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; } }

    // ========== Поллинг статусов сообщений ==========
    let statusPollingInterval = null;
    function startStatusPolling() {
        if (statusPollingInterval) clearInterval(statusPollingInterval);
        // ИСПРАВЛЕНИЕ: увеличен интервал с 8 до 30 сек. WS шлёт message_status мгновенно, поллинг — лишь fallback
        statusPollingInterval = setInterval(async () => {
            const myMessages = document.querySelectorAll('.message-own');
            const ids = Array.from(myMessages)
                .filter(el => el.dataset.status !== 'read' && el.dataset.id && !el.dataset.id.startsWith('temp'))
                .map(el => el.dataset.id);
            if (ids.length === 0) return;
            try {
                const res = await fetch('/message/statuses', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids })
                });
                if (!res.ok) return;
                const statuses = await res.json();
                for (const [id, st] of Object.entries(statuses)) {
                    const msgDiv = document.querySelector(`.message-own[data-id="${id}"]`);
                    if (msgDiv && msgDiv.dataset.status !== st) {
                        msgDiv.dataset.status = st;
                        if (window.updateStatusIcon) window.updateStatusIcon(msgDiv, st);
                        const chatId = State.currentChatAddress;
                        const cached = window.getCachedMessages ? window.getCachedMessages(chatId) : [];
                        const cachedMsg = cached.find(m => String(m.id) === String(id));
                        if (cachedMsg) cachedMsg.status = st;
                    }
                }
            } catch(e) { console.warn('Status polling error', e); }
        }, 30000);
    }
    function stopStatusPolling() { if (statusPollingInterval) { clearInterval(statusPollingInterval); statusPollingInterval = null; } }

    // ========== Поллинг статусов пользователей ==========
    let userStatusPollingInterval = null;
    async function pollUserStatuses() {
        const items = document.querySelectorAll('.conversation-item:not([data-is-group="1"])');
        const addresses = Array.from(items)
            .map(el => el.dataset.address)
            .filter(addr => addr && addr !== State.userAddress);
        if (addresses.length === 0) return;
        if (!window.fetchUserStatuses) return;
        const statuses = await window.fetchUserStatuses(addresses);
        for (const el of items) {
            const addr = el.dataset.address;
            const status = statuses[addr]?.status || 'offline';
            const statusSpan = el.querySelector('.status');
            if (statusSpan) {
                statusSpan.className = `status ${status}`;
                statusSpan.title = status === 'online' ? t('online') : t('offline');
            }
        }
    }
    function startUserStatusPolling() {
        if (userStatusPollingInterval) clearInterval(userStatusPollingInterval);
        userStatusPollingInterval = setInterval(() => { pollUserStatuses(); }, 30000);
    }
    function stopUserStatusPolling() {
        if (userStatusPollingInterval) {
            clearInterval(userStatusPollingInterval);
            userStatusPollingInterval = null;
        }
    }

    // ========== Push Notifications ==========
    window.initPushNotifications = initPushNotifications;
    window.urlBase64ToUint8Array = urlBase64ToUint8Array;

    async function initPushNotifications() {
        if (!('serviceWorker' in navigator)) return;
        if (!('PushManager' in window)) return;

        const permission = Notification.permission;
        if (permission === 'denied') {
            console.log('Push: denied by user');
            return;
        }
        if (permission !== 'granted') {
            console.log('Push: permission not granted, will request after user gesture');
            return;
        }

        try {
            const registration = await navigator.serviceWorker.ready;

            let publicKey;
            try {
                const keyRes = await fetch('/push/vapid-public-key');
                if (keyRes.ok) {
                    const keyData = await keyRes.json();
                    publicKey = keyData.publicKey;
                }
            } catch(e) { /* fallback к хардкоду */ }

            if (!publicKey) {
                publicKey = 'BPa5fghsHcpAbmlQTdXg6WzoMC_iPaDMzFY4mc2BUipmno6sLxN6KoSfaZfgUFkh9c0B34XhBvC93WXn92xKlkw';
            }

            const applicationServerKey = urlBase64ToUint8Array(publicKey);
            let subscription = await registration.pushManager.getSubscription();

            if (subscription) {
                try {
                    const res = await fetch('/push/subscribe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(subscription)
                    });
                    if (res.ok) {
                        console.log('Push: existing subscription refreshed on server ✓');
                    } else {
                        console.error('Push: server rejected existing subscription', res.status);
                        await subscription.unsubscribe();
                        subscription = null;
                    }
                } catch (err) {
                    console.error('Push: failed to refresh subscription', err);
                }
                if (subscription) return;
            }

            try {
                subscription = await registration.pushManager.subscribe({
                    userVisibleOnly: true,
                    applicationServerKey: applicationServerKey
                });
                console.log('Push: new subscription created');
            } catch(e) {
                console.error('Push: subscribe failed', e.name, e.message);
                return;
            }

            const res = await fetch('/push/subscribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(subscription)
            });

            if (res.ok) {
                console.log('Push: subscription synced with server ✓');
            } else {
                console.error('Push: server rejected subscription', res.status);
            }

        } catch(e) {
            console.error('Push init failed', e);
        }
    }

    function urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/\-/g, '+').replace(/_/g, '/');
        const rawData = atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        return outputArray;
    }

    // Обработка сообщений от Service Worker (клик по push-уведомлению)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.addEventListener('message', event => {
            if (event.data?.type === 'navigate' && event.data?.url) {
                const target = event.data.url;
                if (window.location.pathname !== new URL(target, location.origin).pathname) {
                    window.location.href = target;
                } else {
                    window.focus();
                }
            }

            if (event.data?.type === 'pushsubscriptionchange') {
                console.log('Re-subscribing push due to subscription change');
                if (window.initPushNotifications) window.initPushNotifications();
            }

            if (event.data?.type === 'open_call') {
                const callId = event.data.call_id;
                if (callId && !sessionStorage.getItem('pending_call_id')) {
                    sessionStorage.setItem('pending_call_id', callId);
                    window._pendingCallHandled = false;
                }

                if (window.wsClient && window.wsClient.isConnected) {
                    console.log('[App] Requesting call details for', callId);
                    window.wsClient.send({ type: 'get_call', call_id: callId });
                } else {
                    console.log('[App] WebSocket not ready, pending call stored');
                }
            }
        });
    }

    // Экспорт глобальных функций
    window.getPubKey = getPubKey;
    window.compressImage = compressImage;
    window.ensureKeys = ensureKeys;
    window.initWebSocket = initWebSocket;
    window.startHeartbeat = startHeartbeat;
    window.stopHeartbeat = stopHeartbeat;
    window.startStatusPolling = startStatusPolling;
    window.stopStatusPolling = stopStatusPolling;
    window.processMessageDecryption = processMessageDecryption;
    window.handleWebSocketMessage = handleWebSocketMessage;
    window.startUserStatusPolling = startUserStatusPolling;
    window.stopUserStatusPolling = stopUserStatusPolling;
    window.wsClient = null;
})();