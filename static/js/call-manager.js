/**
 * call-manager.js — WebRTC менеджер с поддержкой аудио/видео, восстановлением при переключении вкладок,
 * ICE‑restart на iOS, таймером разговора, i18n и сохранением истории звонков.
 *
 * Исправления:
 *   - Разрешён ICE‑restart на iOS.
 *   - Увеличен таймер _disconnectTimer до 20 секунд.
 *   - Улучшен _visibilityHandler: восстановление WebSocket, переподключение аудио, каскадное восстановление.
 *   - Добавлена поддержка видеозвонков (отображение локального и удалённого видео).
 *   - Исправлено восстановление видео после сворачивания.
 *   - Автоопределение видеозвонка по SDP.
 *   - Динамическое изменение размера модального окна.
 *   - Сохранение истории звонков (входящие/исходящие, статус, длительность).
 */
(function() {
    if (window.CallManagerLoaded) return;
    window.CallManagerLoaded = true;

    class CallManager {
        constructor() {
            this.pc = null;
            this.localStream = null;
            this.currentCallId = null;
            this.currentPartner = null;
            this.currentPartnerName = null;
            this.isInitiator = false;
            this.isAudioOnly = true;         // true = только аудио, false = видео
            this.iceServers = [];
            this.isMuted = false;
            this.isSpeakerEnabled = false;
            this.activeModal = null;
            this.remoteAudio = null;
            this.isCompactMode = false;
            this.isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
            this.outsideClickListener = null;
            this.audioCtx = null;
            this._visibilityHandler = null;
            this._incomingModalObserver = null;

            // Буфер для ICE candidates
            this.pendingCandidates = [];
            this.pendingAnswer = null;

            // Таймер разговора
            this.callStartTime = null;
            this.callDurationInterval = null;

            // Флаг: соединение в процессе установки
            this.isEstablishing = false;
            this._disconnectTimer = null;
            this._connectTimeout = null;

            // Для управления видео
            this.isVideoEnabled = false;      // показывает, включена ли камера в данный момент
            this._initialized = false;

            // ── НОВЫЕ СВОЙСТВА ДЛЯ ИСТОРИИ ЗВОНКОВ ──
            this.lastCallStartTime = null;     // момент установки соединения (для длительности)
            this.currentCallType = null;       // 'audio' или 'video'
            this.currentCallDirection = null;  // 'incoming' или 'outgoing'
        }

        // ========== i18n и таймеры ==========
        t(key, defaultValue = '') {
            if (typeof i18next !== 'undefined' && i18next.isInitialized && i18next.exists(key)) {
                return i18next.t(key);
            }
            const fallback = {
                'call_calling': 'Calling...',
                'call_connected': 'Connected',
                'call_connecting': 'Connecting...',
                'call_accept': 'Accept',
                'call_reject': 'Reject',
                'call_end': 'End call',
                'call_mute_microphone': 'Mute microphone',
                'call_unmute_microphone': 'Unmute microphone',
                'call_speaker': 'Speaker',
                'call_earpiece': 'Earpiece',
                'call_minimize': 'Minimize',
                'call_expand': 'Expand',
                'call_from': 'from',
                'incoming_call': 'Incoming call',
                'call_ended': 'Call ended',
                'call_rejected': 'Call rejected'
            };
            return fallback[key] || defaultValue || key;
        }

        formatDuration(ms) {
            const totalSeconds = Math.floor(ms / 1000);
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const seconds = totalSeconds % 60;
            if (hours > 0) {
                return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
            return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }

        updateDurationDisplay() {
            if (!this.callStartTime) {
                this.setDurationText('00:00');
                return;
            }
            const elapsed = Date.now() - this.callStartTime;
            const formatted = this.formatDuration(elapsed);
            this.setDurationText(formatted);
        }

        setDurationText(text) {
            const durationElem = document.getElementById('callDuration');
            if (durationElem) durationElem.textContent = text;
            const miniDurationElem = document.getElementById('miniDuration');
            if (miniDurationElem) miniDurationElem.textContent = text;
        }

        startCallTimer() {
            if (this.callDurationInterval) clearInterval(this.callDurationInterval);
            this.callStartTime = Date.now();
            this.updateDurationDisplay();
            this.callDurationInterval = setInterval(() => this.updateDurationDisplay(), 1000);
        }

        stopCallTimer() {
            if (this.callDurationInterval) {
                clearInterval(this.callDurationInterval);
                this.callDurationInterval = null;
            }
            this.callStartTime = null;
            this.setDurationText('00:00');
        }

        // ========== Инициализация ==========
        async init() {
            if (this._initialized) {
                console.log('[CallManager] Already initialized, skipping');
                return;
            }
            this._initialized = true;

            try {
                const res = await fetch('/calls/turn-credentials');
                const data = await res.json();
                this.iceServers = data.iceServers;
                console.log('ICE servers loaded', this.iceServers);
                if ("Notification" in window && Notification.permission === "default") {
                    await Notification.requestPermission();
                }
            } catch(e) {
                console.warn('Failed to load TURN config, fallback to STUN only', e);
                this.iceServers = [{ urls: 'stun:stun.l.google.com:19302' }];
            }

            this.remoteAudio = document.createElement('audio');
            this.remoteAudio.id = 'remoteAudio';
            this.remoteAudio.autoplay = true;
            this.remoteAudio.setAttribute('playsinline', '');
            this.remoteAudio.style.position = 'absolute';
            this.remoteAudio.style.visibility = 'hidden';
            document.body.appendChild(this.remoteAudio);

            if (this.isIOS) {
                const unlockGesture = async () => {
                    if (this.remoteAudio) {
                        await this.remoteAudio.play().catch(() => {});
                        await this.unlockAudioContext();
                    }
                    document.removeEventListener('touchstart', unlockGesture);
                    document.removeEventListener('click', unlockGesture);
                };
                document.addEventListener('touchstart', unlockGesture, { once: true });
                document.addEventListener('click', unlockGesture, { once: true });
            }

            // ========== Улучшенный обработчик видимости ==========
            this._visibilityHandler = async () => {
                if (document.visibilityState !== 'visible') return;
                if (!this.pc) return;
                if (this.isEstablishing) {
                    console.log('[CallManager] Skipping visibility recovery – call is establishing');
                    return;
                }

                // Восстанавливаем WebSocket
                if (window.wsClient && !window.wsClient.isConnected && typeof window.initWebSocket === 'function') {
                    console.warn('[CallManager] WebSocket disconnected, reconnecting...');
                    window.initWebSocket().catch(err => console.warn('[CallManager] WS reconnect failed:', err));
                }

                // Всегда переподключаем аудио и видео (на всякий случай)
                this.reattachRemoteStream();
                await this.refreshLocalVideo();

                // Показываем контейнер видео, если нужно
                if (!this.isAudioOnly) {
                    this.showVideoContainer(true);
                } else {
                    this.showVideoContainer(false);
                }
                this.updateModalSize();

                const state = this.pc.connectionState;
                const ice = this.pc.iceConnectionState;

                if (state === 'failed' || state === 'closed' || ice === 'failed') {
    console.warn('[CallManager] Recovery: restarting call due to state', { state, ice });
    // ИСПРАВЛЕНИЕ: для получателя пробуем переотправить answer
    if (!this.isInitiator && this._lastOffer) {
        console.warn('[CallManager] Re-answering call as non-initiator');
        this.answerCall(this.currentCallId, this.currentPartner, this._lastOffer, this.currentPartnerName, !this.isAudioOnly);
        return;
    }
    this.restartCallFull();
} else {
                console.warn('[CallManager] Forcing ICE restart to restore video after tab switch');
                this.restartIce().catch(() => {
        console.warn('[CallManager] ICE restart failed, falling back to full restart');
        // ИСПРАВЛЕНИЕ: аналогично для получателя при неудачном ICE-restart
        if (!this.isInitiator && this._lastOffer) {
            this.answerCall(this.currentCallId, this.currentPartner, this._lastOffer, this.currentPartnerName, !this.isAudioOnly);
            return;
        }
        this.restartCallFull();
    });
                 }
                    };
            document.addEventListener('visibilitychange', this._visibilityHandler);

            if (typeof i18next !== 'undefined' && i18next.isInitialized) {
                this.createMiniWidget();
                this.updateLocalizedTexts();
            } else {
                document.addEventListener('i18next:initialized', () => {
                    this.createMiniWidget();
                    this.updateLocalizedTexts();
                }, { once: true });
            }
        }

        // ========== Работа с аудиоконтекстом (iOS) ==========
        unlockAudioContext() {
            if (!this.isIOS) return Promise.resolve();
            if (this.audioCtx && this.audioCtx.state === 'running') return Promise.resolve();
            try {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const buf = this.audioCtx.createBuffer(1, 1, 22050);
                const src = this.audioCtx.createBufferSource();
                src.buffer = buf;
                src.connect(this.audioCtx.destination);
                src.start(0);
                return this.audioCtx.resume();
            } catch(e) {
                console.warn('[iOS] AudioContext unlock failed:', e);
                return Promise.resolve();
            }
        }

        // ========== Звук входящего вызова ==========
        playIncomingSound() {
            if (this._incomingAudio) return;
            this.unlockAudioContext().then(() => {
                try {
                    const audio = new Audio("/sounds/incoming.mp3");
                    audio.loop = true;
                    audio.volume = 0.8;
                    const playPromise = audio.play();
                    if (playPromise !== undefined) {
                        playPromise.catch(err => console.warn("[CallManager] sound blocked:", err));
                    }
                    this._incomingAudio = audio;
                } catch (e) {
                    console.warn("incoming sound error", e);
                }
            }).catch(() => {});
        }

        stopIncomingSound() {
            if (this._incomingAudio) {
                this._incomingAudio.pause();
                this._incomingAudio.currentTime = 0;
                this._incomingAudio = null;
            }
        }

        // ========== Переподключение удалённого аудио ==========
        reattachRemoteStream() {
            if (!this.remoteAudio || !window._remoteStream) return;
            this.remoteAudio.srcObject = null;
            this.remoteAudio.srcObject = window._remoteStream;
            this.remoteAudio.load();
            this.remoteAudio.play().catch(e => console.warn('[CallManager] remoteAudio play after reattach:', e));
        }

        // ========== Переподключение видео (локального и удалённого) ==========
        reattachVideoStreams() {
            // --- Удалённое видео ---
            if (window._remoteStream) {
                const remoteVideo = document.getElementById('remoteVideo');
                if (remoteVideo) {
                    const videoTracks = window._remoteStream.getVideoTracks();
                    if (videoTracks.length > 0 && videoTracks[0].readyState === 'live') {
                        remoteVideo.srcObject = null;
                        setTimeout(() => {
                            remoteVideo.srcObject = window._remoteStream;
                            remoteVideo.load();
                            remoteVideo.play().catch(e => console.warn('[CallManager] remoteVideo play after reattach:', e));
                        }, 50);
                    } else {
                        console.warn('[CallManager] Remote video track not live, skipping reattach');
                    }
                }
            }

            // --- Локальное видео ---
            if (this.localStream) {
                const localVideo = document.getElementById('localVideo');
                if (localVideo) {
                    const videoTracks = this.localStream.getVideoTracks();
                    if (videoTracks.length > 0 && videoTracks[0].readyState === 'live') {
                        localVideo.srcObject = null;
                        setTimeout(() => {
                            localVideo.srcObject = this.localStream;
                            localVideo.load();
                            localVideo.play().catch(e => console.warn('[CallManager] localVideo play after reattach:', e));
                        }, 50);
                    }
                }
            }
        }

        // ========== Буферизация ICE-кандидатов ==========
        flushPendingCandidates() {
            if (!this.pc) return;
            console.log('[ICE] Flushing', this.pendingCandidates.length, 'buffered candidates');
            for (const candidate of this.pendingCandidates) {
                this.pc.addIceCandidate(new RTCIceCandidate(candidate))
                    .then(() => console.log('[ICE] Buffered candidate added OK'))
                    .catch(err => console.warn('[ICE] Failed to add buffered candidate:', err));
            }
            this.pendingCandidates = [];
        }

        // ========== ICE‑restart (без блокировки iOS) ==========
        async restartIce() {
            if (!this.pc || !this.currentPartner) return;
            try {
                const offer = await this.pc.createOffer({ iceRestart: true });
                offer.sdp = this.preferOpusCodec(offer.sdp);
                await this.pc.setLocalDescription(offer);
                window.wsClient?.send({
                    type: 'call_offer',
                    target: this.currentPartner,
                    call_id: this.currentCallId,
                    sdp: offer,
                    is_restart: true
                });
                console.log('[CallManager] ICE restart offer sent');
            } catch(e) {
                console.error('[CallManager] ICE restart failed:', e);
                await this.restartCallFull();
            }
        }

        // ========== Полный перезапуск звонка (только для инициатора) ==========
        async restartCallFull() {
            if (!this.currentPartner || !this.currentCallId) return;
            if (!this.isInitiator) {
                console.warn('[CallManager] Incoming call lost after sleep – ending call');
                this.endCall();
                window.NotificationManager?.showToast('Call lost due to connection timeout', 'error');
                return;
            }
            const partner = this.currentPartner;
            const isAudioOnly = this.isAudioOnly;
            const partnerName = this.currentPartnerName;
            console.log('[CallManager] FULL RESTART CALL', this.currentCallId);
            this.endCall();
            await new Promise(resolve => setTimeout(resolve, 300));
            await this.makeCall(partner, !isAudioOnly, partnerName);
        }

        // ========== Кодек Opus (для аудио) ==========
        preferOpusCodec(sdp) {
            if (this.isIOS) return sdp;
            if (!sdp) return sdp;
            const lines = sdp.split('\r\n');
            const mIdx = lines.findIndex(l => l.startsWith('m=audio'));
            if (mIdx === -1) return sdp;
            const opusLine = lines.find(l => /opus\/48000/i.test(l));
            if (!opusLine) return sdp;
            const payload = opusLine.match(/:(\d+)/)?.[1];
            if (!payload) return sdp;
            lines[mIdx] = lines[mIdx].replace(
                /^(m=audio \d+ \S+)(.*)/,
                (_, prefix, rest) => {
                    const payloads = rest.trim().split(' ').filter(p => p !== payload);
                    return `${prefix} ${payload} ${payloads.join(' ')}`;
                }
            );
            return lines.join('\r\n');
        }

        // ========== Мини‑виджет (свёрнутый звонок) ==========
        createMiniWidget() {
            if (document.getElementById('callMiniWidget')) return;
            const widget = document.createElement('div');
            widget.id = 'callMiniWidget';
            widget.className = 'call-mini-widget hidden';
            widget.innerHTML = `
                <div class="call-mini-avatar" id="miniAvatar">📞</div>
                <div class="call-mini-info">
                    <div class="call-mini-name" id="miniName">Call</div>
                    <div class="call-mini-status" id="miniStatus"></div>
                    <div class="call-mini-duration" id="miniDuration" style="font-size: 11px; color: var(--text-muted);">00:00</div>
                </div>
                <div class="call-mini-actions">
                    <button class="call-mini-btn" id="miniExpandBtn">⤢</button>
                    <button class="call-mini-btn call-mini-end" id="miniEndBtn">
                        <img src="/static/icons/EndCall.png" width="20" height="20" alt="End call" style="filter: invert(1);">
                    </button>
                </div>
            `;
            document.body.appendChild(widget);
            document.getElementById('miniExpandBtn')?.addEventListener('click', (e) => {
                e.stopPropagation();
                this.expandFromMini();
            });
            document.getElementById('miniEndBtn')?.addEventListener('click', (e) => {
                e.stopPropagation();
                this.endCall();
            });
            widget.addEventListener('click', (e) => {
                if (e.target.closest('.call-mini-btn')) return;
                if (this.isCompactMode) this.expandFromMini();
            });

            // Перетаскивание
            let isDragging = false, startX, startY, offsetX, offsetY;
            const onDragStart = (e, clientX, clientY) => {
                if (e.target.closest('.call-mini-btn')) return;
                isDragging = true;
                startX = clientX; startY = clientY;
                const rect = widget.getBoundingClientRect();
                offsetX = startX - rect.left;
                offsetY = startY - rect.top;
                widget.style.cursor = 'grabbing';
                e.preventDefault();
            };
            const onDragMove = (clientX, clientY) => {
                if (!isDragging) return;
                let left = Math.min(Math.max(clientX - offsetX, 8), window.innerWidth - widget.offsetWidth - 8);
                let top  = Math.min(Math.max(clientY - offsetY, 8), window.innerHeight - widget.offsetHeight - 8);
                widget.style.left = left + 'px';
                widget.style.top = top + 'px';
                widget.style.right = 'auto';
                widget.style.bottom = 'auto';
            };
            const onDragEnd = () => {
                isDragging = false;
                widget.style.cursor = 'grab';
            };
            widget.addEventListener('mousedown', (e) => onDragStart(e, e.clientX, e.clientY));
            document.addEventListener('mousemove', (e) => onDragMove(e.clientX, e.clientY));
            document.addEventListener('mouseup', onDragEnd);
            widget.addEventListener('touchstart', (e) => {
                const t = e.touches[0];
                onDragStart(e, t.clientX, t.clientY);
            }, { passive: false });
            document.addEventListener('touchmove', (e) => {
                if (!isDragging) return;
                const t = e.touches[0];
                onDragMove(t.clientX, t.clientY);
                e.preventDefault();
            }, { passive: false });
            document.addEventListener('touchend', onDragEnd);
        }

        // ========== Клик вне модалки (сворачивание) ==========
        addOutsideClickListener() {
            this.removeOutsideClickListener();
            const attachAfterDelay = () => {
                this.outsideClickListener = (e) => {
                    const modal = document.getElementById('callModal');
                    if (!modal || modal.classList.contains('hidden')) return;
                    if (e.target.closest('#callCollapseBtn')) return;
                    if (e.target.closest('#callMiniWidget')) return;
                    if (!e.target.closest('.modal')) {
                        this.collapseToMini();
                    }
                };
                document.addEventListener('click', this.outsideClickListener);
                document.addEventListener('touchstart', this.outsideClickListener);
            };
            setTimeout(attachAfterDelay, 400);
        }

        removeOutsideClickListener() {
            if (this.outsideClickListener) {
                document.removeEventListener('click', this.outsideClickListener);
                document.removeEventListener('touchstart', this.outsideClickListener);
                this.outsideClickListener = null;
            }
        }

        // ========== Получение медиапотока (аудио + видео) ==========
        async getUserMedia() {
            if (this.localStream) {
                const tracks = this.localStream.getTracks();
                const allLive = tracks.length > 0 && tracks.every(t => t.readyState === 'live');
                if (!allLive) {
                    console.warn('[CallManager] Cached localStream has dead tracks, requesting new');
                    this.localStream.getTracks().forEach(t => t.stop());
                    this.localStream = null;
                }
            }
            if (this.localStream) return this.localStream;

            const constraints = {
                audio: true,
                video: !this.isAudioOnly
            };
            const mediaPromise = navigator.mediaDevices.getUserMedia(constraints);
            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Microphone access timeout (10s)')), 10000)
            );

            try {
                this.localStream = await Promise.race([mediaPromise, timeoutPromise]);
                return this.localStream;
            } catch(e) {
                console.error('Media access denied or timed out', e);
                throw new Error('Cannot access microphone/camera. Please check permissions.');
            }
        }

        // ========== Закрытие медиапотоков ==========
        closeMedia() {
            if (this.localStream) {
                this.localStream.getTracks().forEach(t => t.stop());
                this.localStream = null;
            }
            if (this.remoteAudio?.srcObject) {
                this.remoteAudio.srcObject.getTracks().forEach(t => t.stop());
                this.remoteAudio.srcObject = null;
            }
            if (window._remoteStream) {
                window._remoteStream.getTracks().forEach(t => t.stop());
                window._remoteStream = null;
            }
            // Очищаем видеоэлементы
            const localVideo = document.getElementById('localVideo');
            if (localVideo) { localVideo.srcObject = null; localVideo.pause(); }
            const remoteVideo = document.getElementById('remoteVideo');
            if (remoteVideo) { remoteVideo.srcObject = null; remoteVideo.pause(); }
            this.showVideoContainer(false);
        }

        // ========== Управление отображением видео в модалке ==========
        showVideoContainer(show) {
            const container = document.querySelector('.call-video-container');
            if (container) container.style.display = show ? 'block' : 'none';
            const avatar = document.querySelector('.call-avatar');
            if (avatar) avatar.style.display = show ? 'none' : 'block';
            this.updateModalSize();
        }

        // ========== Изменение размера модального окна ==========
        updateModalSize() {
            const modal = document.querySelector('#callModal .call-modal');
            if (!modal) return;

            if (!this.isAudioOnly) {
                if (window.innerWidth < 768) {
                    modal.style.maxWidth = '95vw';
                    modal.style.width = '95%';
                } else {
                    modal.style.maxWidth = '70vw';
                    modal.style.width = '70%';
                }
            } else {
                modal.style.maxWidth = '340px';
                modal.style.width = '90%';
            }
        }

        // ========== Включить/выключить локальную камеру ==========
        toggleVideo() {
            if (!this.localStream) return;
            const videoTracks = this.localStream.getVideoTracks();
            if (videoTracks.length === 0) {
                window.NotificationManager?.showToast('No video track available', 'error');
                return;
            }
            const enabled = videoTracks[0].enabled;
            videoTracks.forEach(t => t.enabled = !enabled);
            this.isVideoEnabled = !enabled;
            const videoBtn = document.getElementById('callVideoToggleBtn');
            if (videoBtn) {
                const videoIcon = document.getElementById('videoIcon');
                if (videoIcon) {
                    videoIcon.src = this.isVideoEnabled ? '/static/icons/Video.png' : '/static/icons/NoVideo.png';
                }
                videoBtn.classList.toggle('active', this.isVideoEnabled);
                videoBtn.title = this.isVideoEnabled ? 'Turn off camera' : 'Turn on camera';
            }
        }

        // ========== Перезапрос локального видео и замена треков ==========
        async refreshLocalVideo() {
            if (this.isAudioOnly) return;
            if (!this.currentPartner || !this.currentCallId) return;

            try {
                const newStream = await navigator.mediaDevices.getUserMedia({
                    audio: true,
                    video: { facingMode: 'user' }
                });

                const newVideoTrack = newStream.getVideoTracks()[0];
                if (!newVideoTrack) {
                    console.warn('[CallManager] No video track in new stream');
                    return;
                }

                const oldVideoTrack = this.localStream?.getVideoTracks()[0];
                if (oldVideoTrack) {
                    const sender = this.pc?.getSenders().find(s => s.track === oldVideoTrack);
                    if (sender) {
                        await sender.replaceTrack(newVideoTrack);
                        console.log('[CallManager] Video track replaced in PeerConnection');
                    } else {
                        this.pc?.addTrack(newVideoTrack, this.localStream);
                    }
                    oldVideoTrack.stop();
                } else {
                    this.pc?.addTrack(newVideoTrack, this.localStream);
                }

                const audioTracks = this.localStream?.getAudioTracks() || [];
                const newStreamWithAudio = new MediaStream([...audioTracks, newVideoTrack]);
                this.localStream = newStreamWithAudio;

                const localVideo = document.getElementById('localVideo');
                if (localVideo) {
                    localVideo.srcObject = null;
                    localVideo.srcObject = this.localStream;
                    localVideo.load();
                    localVideo.play().catch(e => console.warn);
                }

                await this.restartIce();

                window.NotificationManager?.showToast('Camera reconnected', 'info');
            } catch (err) {
                console.error('[CallManager] Failed to refresh local video:', err);
                window.NotificationManager?.showToast('Camera reconnection failed', 'error');
            }
        }

        // ========== Создание исходящего звонка ==========
        async makeCall(partnerAddress, isVideo = false, partnerName = '') {
            if (this.currentCallId) {
                const confirmEnd = confirm(this.t('call_active_message', 'You are already in a call. End it to start a new one?'));
                if (!confirmEnd) {
                    window.NotificationManager?.showToast(this.t('call_in_progress', 'Call already in progress'), 'warning');
                    return;
                }
                this.endCall();
                await new Promise(r => setTimeout(r, 500));
            }

            await this.unlockAudioContext();

            this.isEstablishing = true;
            try {
                this.isAudioOnly = !isVideo;
                this.currentPartner = partnerAddress;
                this.currentPartnerName = partnerName || partnerAddress.slice(0,10) + '…';
                this.isInitiator = true;
                this.currentCallId = `call_${Date.now()}_${Math.random().toString(36)}`;
                this.currentCallDirection = 'outgoing';
                this.currentCallType = isVideo ? 'video' : 'audio';
                this.lastCallStartTime = null; // будет установлен при соединении

                const stream = await this.getUserMedia();
                this.createPeerConnection();
                stream.getTracks().forEach(track => this.pc.addTrack(track, stream));

                const localVideo = document.getElementById('localVideo');
                if (localVideo && !this.isAudioOnly) {
                    localVideo.srcObject = stream;
                    localVideo.play().catch(e => console.warn);
                }
                this.showVideoContainer(!this.isAudioOnly);

                const offer = await this.pc.createOffer();
                offer.sdp = this.preferOpusCodec(offer.sdp);
                await this.pc.setLocalDescription(offer);
                this.flushPendingAnswer();

                window.wsClient?.send({
                    type: 'call_offer',
                    target: partnerAddress,
                    call_id: this.currentCallId,
                    sdp: offer,
                    video: isVideo
                });

                this.showCallModal('outgoing', this.t('call_calling'));
            } catch(err) {
                console.error(err);
                this.endCall();
                window.NotificationManager?.showToast('Cannot start call: ' + err.message, 'error');
            } finally {
                setTimeout(() => { if (this.isEstablishing) this.isEstablishing = false; }, 5000);
            }
        }

        // ========== Создание PeerConnection (с увеличенным таймером) ==========
        createPeerConnection() {
            if (this.pc) {
                this.pc.onconnectionstatechange = null;
                this.pc.oniceconnectionstatechange = null;
                this.pc.onicecandidate = null;
                this.pc.ontrack = null;
                this.pc.close();
                this.stopCallTimer();
            }
            this.pendingAnswer = null;
            window._remoteStream = null;
            if (window._pendingCallHandled) window._pendingCallHandled = false;
            this.pc = new RTCPeerConnection({ iceServers: this.iceServers });

            this.pc.onicecandidate = (event) => {
                if (event.candidate && this.currentPartner) {
                    console.log('[ICE] Sending candidate to', this.currentPartner.slice(0,10), ':', event.candidate.candidate?.split(' ')[7]);
                    window.wsClient?.send({
                        type: 'call_ice',
                        target: this.currentPartner,
                        call_id: this.currentCallId,
                        candidate: event.candidate
                    });
                } else if (!event.candidate) {
                    console.log('[ICE] Gathering complete (null candidate)');
                }
            };

            this.pc.onicegatheringstatechange = () => {
                console.log('[ICE] Gathering state:', this.pc.iceGatheringState);
            };

            this.pc.ontrack = (event) => {
                window._remoteStream = event.streams[0] || (event.track && new MediaStream([event.track]));
                if (this.remoteAudio) {
                    this.remoteAudio.srcObject = null;
                    this.remoteAudio.srcObject = window._remoteStream;
                    this.remoteAudio.muted = true;
                    this.remoteAudio.play()
                        .then(() => { if (this.remoteAudio) this.remoteAudio.muted = false; })
                        .catch(e => { console.warn('Audio play failed:', e); if (this.remoteAudio) this.remoteAudio.muted = false; });
                }
                this.attachRemoteStream(window._remoteStream);
                this.updateCallStatus('connected');

                const hasVideo = window._remoteStream && window._remoteStream.getVideoTracks().length > 0;
                if (hasVideo) {
                    this.showVideoContainer(true);
                    const remoteVideo = document.getElementById('remoteVideo');
                    if (remoteVideo) {
                        remoteVideo.srcObject = window._remoteStream;
                        remoteVideo.play().catch(e => console.warn);
                    }
                }
            };

            this.pc.onconnectionstatechange = () => {
                const s = this.pc.connectionState;
                console.log('[CallManager] connection state:', s, '| ICE:', this.pc.iceConnectionState, '| gathering:', this.pc.iceGatheringState);
                if (s === 'failed' || s === 'closed') {
                    this.endCall();
                } else if (s === 'connecting' || s === 'new') {
                    this.updateCallStatus('connecting');
                } else if (s === 'connected') {
                    if (this._connectTimeout) { clearTimeout(this._connectTimeout); this._connectTimeout = null; }
                    this.isEstablishing = false;
                    // --- СОХРАНЯЕМ ВРЕМЯ СОЕДИНЕНИЯ ДЛЯ ДЛИТЕЛЬНОСТИ ---
                    this.lastCallStartTime = Date.now();

                    const callModal = document.getElementById('callModal');
                    if (callModal && callModal.classList.contains('hidden') && !this.isCompactMode) {
                        this.showCallModal('active', this.t('call_connected'));
                    } else {
                        this.updateCallStatus('connected');
                    }
                    if (!this.callStartTime) this.startCallTimer();
                }
            };

            this.pc.oniceconnectionstatechange = () => {
                const s = this.pc.iceConnectionState;
                console.log('[CallManager] ICE state:', s);
                if (s === 'failed') {
                    if (this.currentPartner && this.currentCallId) {
                        console.warn('[CallManager] ICE failed, attempting restart');
                        this.restartIce();
                    } else {
                        this.endCall();
                    }
                } else if (s === 'disconnected') {
                    if (this._disconnectTimer) clearTimeout(this._disconnectTimer);
                    this._disconnectTimer = setTimeout(() => {
                        if (this.isEstablishing) {
                            console.log('[CallManager] Skipping disconnect recovery – call is establishing');
                            return;
                        }
                        if (this.pc && this.pc.iceConnectionState === 'disconnected' && this.currentPartner) {
                            console.warn('[CallManager] ICE disconnected too long, restarting call');
                            this.restartCallFull();
                        }
                    }, 20000);
                } else if (s === 'connected' || s === 'completed') {
                    if (this._disconnectTimer) {
                        clearTimeout(this._disconnectTimer);
                        this._disconnectTimer = null;
                    }
                }
            };

            if (this._connectTimeout) clearTimeout(this._connectTimeout);
            this._connectTimeout = setTimeout(() => {
                if (this.pc && this.pc.connectionState !== 'connected') {
                    console.error('[CallManager] Connection timeout');
                    window.NotificationManager?.showToast('Call failed: connection timeout', 'error');
                    this.endCall();
                }
            }, 30000);
        }

        // ========== Прикрепление удалённого видеопотока ==========
        attachRemoteStream(stream) {
            const remoteVideo = document.getElementById('remoteVideo');
            if (remoteVideo && stream.getVideoTracks().length > 0) {
                remoteVideo.srcObject = stream;
                remoteVideo.play().catch(e => console.warn);
                this.showVideoContainer(true);
            }
        }

        // ========== Ответ на входящий звонок ==========
        async answerCall(callId, fromAddress, offerSdp, partnerName = '', isVideo = false) {
            console.log('[answerCall] start, callId:', callId, 'from:', fromAddress);
            this.stopIncomingSound();
            await this.unlockAudioContext();

            this.isEstablishing = true;
            try {
                this.currentCallId = callId;
                this.currentPartner = fromAddress;
                this.currentPartnerName = partnerName || fromAddress.slice(0,10) + '…';
                this.isInitiator = false;
                this.isAudioOnly = !isVideo;
                this.currentCallDirection = 'incoming';
                this.currentCallType = isVideo ? 'video' : 'audio';
                this.lastCallStartTime = null; // будет установлен при соединении

                let offer = offerSdp;
                if (typeof offerSdp === 'string') {
                    try { offer = JSON.parse(offerSdp); } catch(e) { offer = offerSdp; }
                }
                if (!offer || !offer.sdp) {
                    console.error('[answerCall] Invalid or missing offerSdp:', offerSdp);
                    window.NotificationManager?.showToast('Call failed: missing offer SDP', 'error');
                    return;
                }

                const stream = await this.getUserMedia();
                this.createPeerConnection();
                stream.getTracks().forEach(track => this.pc.addTrack(track, stream));

                const localVideo = document.getElementById('localVideo');
                if (localVideo && !this.isAudioOnly) {
                    localVideo.srcObject = stream;
                    localVideo.play().catch(e => console.warn);
                }
                this.showVideoContainer(!this.isAudioOnly);

                this.showCallModal('active', this.t('call_connecting'));

                await this.pc.setRemoteDescription(new RTCSessionDescription(offer));
                this.flushPendingCandidates();

                const answer = await this.pc.createAnswer();
                answer.sdp = this.preferOpusCodec(answer.sdp);
                await this.pc.setLocalDescription(answer);

                window.wsClient?.send({
                    type: 'call_answer',
                    target: fromAddress,
                    call_id: callId,
                    sdp: answer
                });
                console.log('[answerCall] done, waiting for ICE/connection...');
            } catch(err) {
                console.error('[answerCall] error:', err);
                this.endCall();
                window.NotificationManager?.showToast('Cannot answer call: ' + err.message, 'error');
            } finally {
                setTimeout(() => { if (this.isEstablishing) this.isEstablishing = false; }, 5000);
            }
        }

        // ========== Обработка ответа (SDP) ==========
        handleRemoteAnswer(answerSdp) {
            if (!this.pc || !this.pc.localDescription) {
                console.warn('[CallManager] handleRemoteAnswer: pc not ready, buffering answer');
                this.pendingAnswer = answerSdp;
                return;
            }
            this.pc.setRemoteDescription(new RTCSessionDescription(answerSdp))
                .then(() => {
                    console.log('[CallManager] Remote answer set successfully');
                    this.flushPendingCandidates();
                })
                .catch(err => console.error('[CallManager] setRemoteDescription(answer) failed:', err));
        }

        flushPendingAnswer() {
            if (!this.pendingAnswer || !this.pc) return;
            const ans = this.pendingAnswer;
            this.pendingAnswer = null;
            console.log('[CallManager] Flushing buffered answer');
            this.handleRemoteAnswer(ans);
        }

        // ========== Обработка ICE-кандидатов ==========
        handleRemoteIce(candidate) {
            if (!candidate) return;
            if (!this.pc || this.pc.signalingState === 'closed') {
                console.warn('[ICE] PeerConnection closed or closing, ignoring candidate');
                return;
            }
            const candidateInit = (typeof candidate === 'object' && 'candidate' in candidate)
                ? candidate
                : { candidate };
            console.log('[ICE] Received remote candidate:', candidateInit.candidate?.split(' ')[7], '| pc ready:', !!this.pc, '| remoteDesc:', !!this.pc?.remoteDescription);
            if (!this.pc || !this.pc.remoteDescription) {
                this.pendingCandidates.push(candidateInit);
                console.log('[ICE] Buffered (no remoteDescription yet), total:', this.pendingCandidates.length);
                return;
            }
            this.pc.addIceCandidate(new RTCIceCandidate(candidateInit))
                .then(() => console.log('[ICE] Remote candidate added OK'))
                .catch(err => console.error('[ICE] addIceCandidate failed:', err, candidateInit));
        }

        // ========== Управление микрофоном ==========
        toggleMute() {
            if (!this.localStream) return;
            this.isMuted = !this.isMuted;
            this.localStream.getAudioTracks().forEach(track => track.enabled = !this.isMuted);
            const muteBtn = document.getElementById('callMuteBtn');
            if (muteBtn) {
                const muteIcon = document.getElementById('muteIcon');
                if (muteIcon) {
                    muteIcon.src = this.isMuted ? '/static/icons/Mic-off.png' : '/static/icons/Mic.png';
                }
                muteBtn.classList.toggle('active', this.isMuted);
                muteBtn.title = this.t(this.isMuted ? 'call_unmute_microphone' : 'call_mute_microphone');
            }
        }

        // ========== Переключение динамика ==========
        async toggleSpeaker() {
            if (this.isIOS) {
                window.NotificationManager?.showToast('Speaker routing is automatic on iOS', 'info');
                return;
            }
            const audioElem = this.remoteAudio;
            if (!audioElem?.srcObject) {
                window.NotificationManager?.showToast('No active audio stream', 'error');
                return;
            }
            if (!audioElem.setSinkId) {
                window.NotificationManager?.showToast('Speaker switching not supported', 'error');
                return;
            }
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const outputs = devices.filter(d => d.kind === 'audiooutput');
                if (outputs.length < 2) {
                    window.NotificationManager?.showToast('Only one audio output available', 'info');
                    return;
                }
                await audioElem.setSinkId(this.isSpeakerEnabled ? outputs[0].deviceId : outputs[1].deviceId);
                this.isSpeakerEnabled = !this.isSpeakerEnabled;
                const speakerBtn = document.getElementById('callSpeakerBtn');
                if (speakerBtn) {
                    speakerBtn.innerHTML = this.isSpeakerEnabled ? '📢' : '🎧';
                    speakerBtn.classList.toggle('active', this.isSpeakerEnabled);
                    speakerBtn.title = this.t(this.isSpeakerEnabled ? 'call_earpiece' : 'call_speaker');
                }
            } catch(e) {
                console.warn('setSinkId error', e);
            }
        }

        // ========== Повторное соединение (для входящих) ==========
        reconnectCall() {
            if (!this.lastCallId || !this.lastFrom || !this.lastOffer) {
                window.NotificationManager?.showToast('No call to reconnect', 'error');
                return;
            }
            console.log('[CallManager] Reconnecting call', this.lastCallId);
            this.hideIncomingModal();
            this.answerCall(this.lastCallId, this.lastFrom, this.lastOffer, this.lastFromName);
        }

        // ======================================================================
        // ========== НОВЫЙ МЕТОД – СОХРАНЕНИЕ ИСТОРИИ ЗВОНКОВ ==================
        // ======================================================================
        async _saveCallHistory(address, name, direction, status, duration) {
            try {
                const res = await fetch('/calls/log', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        address,          // адрес собеседника
                        name,             // его имя (если есть)
                        direction,        // 'incoming' или 'outgoing'
                        status,           // 'answered', 'missed', 'rejected'
                        duration,         // длительность в секундах
                        timestamp: Math.floor(Date.now() / 1000)
                    })
                });
                if (!res.ok) console.warn('Failed to save call history:', await res.text());
            } catch (e) {
                console.warn('Error saving call history:', e);
            }
        }

        // ========== Завершение звонка (с сохранением истории) ==========
        endCall() {
            // --- СОХРАНЯЕМ ИСТОРИЮ ПЕРЕД ОСТАНОВКОЙ ВСЕХ ПОТОКОВ ---
            if (this.currentPartner && this.currentCallId) {
                // Определяем статус звонка
                let status = 'missed';
                if (this.pc && this.pc.connectionState === 'connected') {
                    status = 'answered';
                } else if (this.pc && (this.pc.connectionState === 'closed' || this.pc.connectionState === 'failed')) {
                    // Если звонок был закрыт до соединения – rejected, но для исходящих лучше missed
                    status = (this.isInitiator && this.pc.connectionState === 'closed') ? 'rejected' : 'missed';
                } else if (this.pc && this.pc.connectionState === 'disconnected') {
                    status = 'missed';
                }

                // Длительность (если есть время соединения)
                const duration = this.lastCallStartTime ? Math.floor((Date.now() - this.lastCallStartTime) / 1000) : 0;
                const direction = this.isInitiator ? 'outgoing' : 'incoming';
                const contactAddress = this.currentPartner;
                const contactName = this.currentPartnerName || contactAddress;

                // Отправляем историю (асинхронно, не блокируем завершение)
                this._saveCallHistory(contactAddress, contactName, direction, status, duration);
            }

            // --- ОСТАЛЬНОЙ КОД ЗАВЕРШЕНИЯ ---
            this.stopIncomingSound();
            this.removeOutsideClickListener();
            this.stopCallTimer();
            if (this._connectTimeout) { clearTimeout(this._connectTimeout); this._connectTimeout = null; }
            if (this._disconnectTimer) { clearTimeout(this._disconnectTimer); this._disconnectTimer = null; }
            if (this.pc) {
                this.pc.onconnectionstatechange = null;
                this.pc.oniceconnectionstatechange = null;
                this.pc.onicegatheringstatechange = null;
                this.pc.onicecandidate = null;
                this.pc.ontrack = null;
                this.pc.close();
                this.pc = null;
            }
            this.closeMedia();
            this.pendingCandidates = [];
            this.pendingAnswer = null;
            if (this.currentPartner && this.currentCallId) {
                window.wsClient?.send({ type: 'call_hangup', target: this.currentPartner, call_id: this.currentCallId });
            }
            this.currentPartner = null;
            this.currentCallId = null;
            this.hideCallModal();
            this.hideMiniWidget();
            this.hideIncomingModal();
            this.updateCallStatus('ended');
            this.isCompactMode = false;
            this.isEstablishing = false;
            this.isVideoEnabled = false;
            // Сбрасываем время соединения
            this.lastCallStartTime = null;
            window.NotificationManager?.showToast(this.t('call_ended'), 'info');
        }

        // ========== Отклонение входящего звонка ==========
        rejectCall(callId, from) {
            window.wsClient?.send({ type: 'call_reject', target: from, call_id: callId });
            this.hideIncomingModal();
            window.NotificationManager?.showToast(this.t('call_rejected'), 'info');
        }

        // ========== Отображение модального окна звонка ==========
        showCallModal(state, statusText = '') {
            let modal = document.getElementById('callModal');
            if (!modal) return;
            if (!modal.classList.contains('hidden') && this.currentCallId && modal.dataset.callId === this.currentCallId) {
                console.log('[CallManager] Call modal already visible, skipping');
                return;
            }
            if (this.currentCallId) modal.dataset.callId = this.currentCallId;

            if (!document.getElementById('callDuration')) {
                const statusDiv = modal.querySelector('.call-status');
                if (statusDiv) {
                    const durationDiv = document.createElement('div');
                    durationDiv.id = 'callDuration';
                    durationDiv.className = 'call-duration';
                    durationDiv.style.marginTop = '8px';
                    durationDiv.style.fontSize = '14px';
                    durationDiv.style.fontWeight = '500';
                    statusDiv.after(durationDiv);
                }
            }

            modal.classList.remove('hidden');
            this.activeModal = 'call';
            this.isCompactMode = false;
            this.hideMiniWidget();
            this.addOutsideClickListener();

            const avatarEl = document.getElementById('callAvatar');
            const nameEl   = document.getElementById('callPartnerName');
            const statusEl = document.getElementById('callStatusText');
            if (nameEl)   nameEl.textContent   = this.currentPartnerName || this.currentPartner || 'Unknown';
            if (avatarEl) avatarEl.textContent  = (this.currentPartnerName?.[0] || '?').toUpperCase();
            if (statusEl) statusEl.textContent  = statusText || (state === 'outgoing' ? this.t('call_calling') : this.t('call_connected'));

            const muteBtn     = document.getElementById('callMuteBtn');
            const speakerBtn  = document.getElementById('callSpeakerBtn');
            const endBtn      = document.getElementById('callEndBtn');
            const collapseBtn = document.getElementById('callCollapseBtn');
            const videoToggleBtn = document.getElementById('callVideoToggleBtn');

            if (muteBtn) {
                muteBtn.innerHTML = '<img src="/static/icons/Mic.png" width="28" height="28" alt="Mute" id="muteIcon" style="filter: invert(1);">';
                muteBtn.classList.remove('active');
                muteBtn.onclick = () => this.toggleMute();
                muteBtn.title = this.t('call_mute_microphone');
            }
            if (speakerBtn) {
                if (this.isIOS) {
                    speakerBtn.style.display = 'none';
                } else {
                    speakerBtn.style.display = '';
                    speakerBtn.innerHTML = '🎧';
                    speakerBtn.classList.remove('active');
                    speakerBtn.onclick = () => this.toggleSpeaker();
                    speakerBtn.title = this.t('call_speaker');
                }
            }
            if (endBtn) {
                endBtn.innerHTML = '<img src="/static/icons/EndCall.png" width="28" height="28" alt="Call" style="filter: invert(1);">';
                endBtn.onclick = () => this.endCall();
                endBtn.title = this.t('call_end');
            }
            if (collapseBtn) {
                collapseBtn.onclick = () => this.collapseToMini();
                collapseBtn.title = this.t('call_minimize');
            }

            if (videoToggleBtn) {
                if (this.isAudioOnly) {
                    videoToggleBtn.style.display = 'none';
                } else {
                    videoToggleBtn.style.display = '';
                    videoToggleBtn.innerHTML = '<img src="/static/icons/Video.png" width="24" height="24" alt="Video" id="videoIcon" style="filter: invert(1);">';
                    videoToggleBtn.classList.remove('active');
                    videoToggleBtn.onclick = () => this.toggleVideo();
                    videoToggleBtn.title = 'Turn off camera';
                }
            }

            this.isMuted = false;
            this.isSpeakerEnabled = false;
            this.isVideoEnabled = !this.isAudioOnly;
            this.updateLocalizedTexts();
            this.updateModalSize();

            // Восстанавливаем воспроизведение видео при разворачивании
            const remoteVideo = document.getElementById('remoteVideo');
            if (remoteVideo && remoteVideo.srcObject) {
                remoteVideo.play().catch(e => console.warn('[CallManager] remoteVideo play after expand:', e));
            }
            const localVideo = document.getElementById('localVideo');
            if (localVideo && localVideo.srcObject) {
                localVideo.play().catch(e => console.warn('[CallManager] localVideo play after expand:', e));
            }
        }

        // ========== Сворачивание в мини‑виджет ==========
        collapseToMini() {
            if (this.isCompactMode) return;
            this.isCompactMode = true;
            this.hideCallModal();
            this.showMiniWidget();
            this.removeOutsideClickListener();
        }

        expandFromMini() {
            if (!this.isCompactMode) return;
            this.isCompactMode = false;
            this.hideMiniWidget();
            this.showCallModal('active', this.t('call_connected'));
        }

        showMiniWidget() {
            const widget = document.getElementById('callMiniWidget');
            if (!widget) return;
            widget.classList.remove('hidden');
            const nameSpan = document.getElementById('miniName');
            const statusSpan = document.getElementById('miniStatus');
            if (nameSpan)   nameSpan.textContent   = this.currentPartnerName || 'Call';
            if (statusSpan) statusSpan.textContent  = this.t('call_connected');
            this.updateLocalizedTexts();
        }

        hideMiniWidget() {
            document.getElementById('callMiniWidget')?.classList.add('hidden');
        }

        // ========== Обновление статуса в интерфейсе ==========
        updateCallStatus(status) {
            const statusEl = document.getElementById('callStatusText');
            if (statusEl) {
                if (status === 'connected')        statusEl.textContent = this.t('call_connected');
                else if (status === 'calling')     statusEl.textContent = this.t('call_calling');
                else if (status === 'connecting')  statusEl.textContent = this.t('call_connecting');
                else                               statusEl.textContent = this.t('call_connecting');
            }
            const miniStatus = document.getElementById('miniStatus');
            if (miniStatus) {
                miniStatus.textContent = status === 'connected' ? this.t('call_connected') : this.t('call_connecting');
            }
        }

        hideCallModal() {
            const modal = document.getElementById('callModal');
            if (modal) modal.classList.add('hidden');
            this.activeModal = null;
        }

        // ========== Модальное окно входящего звонка ==========
        showIncomingCallModal(callId, from, offerSdp, fromName = '', bufferedCandidates = [], isVideo = false) {
            this.playIncomingSound();
            const modal = document.getElementById('incomingCallModal');
            if (!modal) return;
            if (this._incomingModalObserver) {
                this._incomingModalObserver.disconnect();
                this._incomingModalObserver = null;
            }

            // Автоопределение видео по SDP
            let isVideoFlag = isVideo;
            if (!isVideoFlag && offerSdp) {
                const sdpString = typeof offerSdp === 'object' && offerSdp.sdp ? offerSdp.sdp : offerSdp;
                if (typeof sdpString === 'string') {
                    isVideoFlag = sdpString.includes('m=video');
                    if (isVideoFlag) {
                        console.log('[CallManager] Video detected from SDP (m=video found)');
                    }
                }
            }

            modal.classList.remove('hidden');
            this.activeModal = 'incoming';
            modal.dataset.callId = callId;
            modal.dataset.from = from;
            modal.dataset.offerSdp = JSON.stringify(offerSdp);
            modal.dataset.video = isVideoFlag ? 'true' : 'false';
            this.currentPartnerName = fromName || from.slice(0,16) + '…';
            // ИСПРАВЛЕНИЕ: сохраняем оффер для возможного повторного ответа
            this._lastOffer = offerSdp;

            this.pendingCandidates = bufferedCandidates || [];
            console.log('[CallManager] Buffered candidates from server:', this.pendingCandidates.length);

            const nameSpan = document.getElementById('incomingCallerName');
            if (nameSpan) nameSpan.textContent = this.currentPartnerName;

            const acceptBtn = document.getElementById('acceptCallBtn');
            const rejectBtn = document.getElementById('rejectCallBtn');

            if (acceptBtn) {
                acceptBtn.innerHTML = '<img src="/static/icons/AddCall.png" width="28" height="28" alt="Accept" style="filter: invert(1);">';
                acceptBtn.title = this.t('call_accept');
                acceptBtn.onclick = async () => {
                    await this.unlockAudioContext();
                    let actualOffer = offerSdp;
                    if (!actualOffer || (typeof actualOffer === 'object' && !actualOffer?.sdp)) {
                        const raw = modal.dataset.offerSdp;
                        if (raw) {
                            try { actualOffer = JSON.parse(raw); } catch(e) { actualOffer = null; }
                        }
                    }
                    const actualCallId   = modal.dataset.callId   || callId;
                    const actualFrom     = modal.dataset.from      || from;
                    const videoFlag = modal.dataset.video === 'true';
                    this.hideIncomingModal();
                    this.answerCall(actualCallId, actualFrom, actualOffer, fromName, videoFlag);
                };
            }
            if (rejectBtn) {
                rejectBtn.innerHTML = '<img src="/static/icons/EndCall.png" width="28" height="28" alt="Reject" style="filter: invert(1);">';
                rejectBtn.title = this.t('call_reject');
                rejectBtn.onclick = () => {
                    this.rejectCall(callId, from);
                    this.hideIncomingModal();
                };
            }

            const fromLabel = modal.querySelector('.call-status');
            if (fromLabel) {
                fromLabel.innerHTML = `<span>${this.t('call_from')}</span> <strong id="incomingCallerName">${this.currentPartnerName}</strong>`;
            }
            const incomingTitle = modal.querySelector('.call-name');
            if (incomingTitle) incomingTitle.textContent = this.t('incoming_call');

            this.updateLocalizedTexts();

            setTimeout(() => {
                if (!modal.classList.contains('hidden') && !this._incomingModalObserver) {
                    this._incomingModalObserver = new MutationObserver((mutations) => {
                        for (const m of mutations) {
                            if (m.attributeName === 'class' && modal.classList.contains('hidden')) {
                                if (this.activeModal === 'incoming') {
                                    console.warn('[CallManager] Incoming modal was hidden externally, showing again');
                                    modal.classList.remove('hidden');
                                }
                            }
                        }
                    });
                    this._incomingModalObserver.observe(modal, { attributes: true });
                }
            }, 50);
        }

        // ========== Push-уведомление о входящем звонке ==========
        showIncomingNotification(name, from) {
            if (!("Notification" in window)) return;
            const title = "📞 Incoming call";
            const options = {
                body: name || from,
                tag: "incoming-call",
                renotify: true,
                silent: false
            };
            if (Notification.permission === "granted") {
                const n = new Notification(title, options);
                n.onclick = () => {
                    window.focus();
                    this.showIncomingCallModal?.();
                };
            } else if (Notification.permission !== "denied") {
                Notification.requestPermission();
            }
        }

        hideIncomingModal() {
            if (this.activeModal === 'incoming') this.activeModal = null;
            if (this._incomingModalObserver) {
                this._incomingModalObserver.disconnect();
                this._incomingModalObserver = null;
            }
            const modal = document.getElementById('incomingCallModal');
            if (modal) modal.classList.add('hidden');
        }

        // ========== Обновление локализованных надписей ==========
        updateLocalizedTexts() {
            const miniStatus = document.getElementById('miniStatus');
            if (miniStatus) miniStatus.textContent = this.t('call_connected');
            const miniExpand = document.getElementById('miniExpandBtn');
            if (miniExpand) miniExpand.title = this.t('call_expand');
            const miniEnd = document.getElementById('miniEndBtn');
            if (miniEnd) miniEnd.title = this.t('call_end');

            const callModal = document.getElementById('callModal');
            if (callModal && !callModal.classList.contains('hidden')) {
                const statusEl = document.getElementById('callStatusText');
                if (statusEl) {
                    const cs = this.pc?.connectionState;
                    if (cs === 'connected') statusEl.textContent = this.t('call_connected');
                    else if (cs === 'connecting') statusEl.textContent = this.t('call_connecting');
                    else statusEl.textContent = this.t('call_calling');
                }
                const muteBtn = document.getElementById('callMuteBtn');
                if (muteBtn) muteBtn.title = this.t(this.isMuted ? 'call_unmute_microphone' : 'call_mute_microphone');
                const speakerBtn = document.getElementById('callSpeakerBtn');
                if (speakerBtn && !this.isIOS) speakerBtn.title = this.t(this.isSpeakerEnabled ? 'call_earpiece' : 'call_speaker');
                const endBtn = document.getElementById('callEndBtn');
                if (endBtn) endBtn.title = this.t('call_end');
                const collapseBtn = document.getElementById('callCollapseBtn');
                if (collapseBtn) collapseBtn.title = this.t('call_minimize');
                const videoToggleBtn = document.getElementById('callVideoToggleBtn');
                if (videoToggleBtn && !this.isAudioOnly) {
                    videoToggleBtn.title = this.isVideoEnabled ? 'Turn off camera' : 'Turn on camera';
                }
            }

            const incomingModal = document.getElementById('incomingCallModal');
            if (incomingModal && !incomingModal.classList.contains('hidden')) {
                const fromLabel = incomingModal.querySelector('.call-status');
                if (fromLabel && this.currentPartnerName) {
                    fromLabel.innerHTML = `<span>${this.t('call_from')}</span> <strong>${this.currentPartnerName}</strong>`;
                }
                const incomingTitle = incomingModal.querySelector('.call-name');
                if (incomingTitle) incomingTitle.textContent = this.t('incoming_call');
                const acceptBtn = document.getElementById('acceptCallBtn');
                if (acceptBtn) acceptBtn.title = this.t('call_accept');
                const rejectBtn = document.getElementById('rejectCallBtn');
                if (rejectBtn) rejectBtn.title = this.t('call_reject');
            }

            document.querySelectorAll('#callModal [data-i18n-title], #incomingCallModal [data-i18n-title], #callMiniWidget [data-i18n-title]').forEach(el => {
                const key = el.getAttribute('data-i18n-title');
                if (key) el.title = this.t(key);
            });
        }
    }

    // ========== Создание и инициализация глобального экземпляра ==========
    window.CallManager = new CallManager();
    window.CallManager.init();

    // ========== Глобальный обработчик сигналов WebSocket ==========
    window.handleCallSignal = (data) => {
        const { type, call_id, from, sdp, candidate, from_name, candidates, video } = data;

        if (call_id && window.CallManager.currentCallId && call_id !== window.CallManager.currentCallId) {
            console.warn('[CallManager] stale signal ignored. current:', window.CallManager.currentCallId, 'received:', call_id);
            return;
        }

        if (type === 'incoming_call') {
            if (window.CallManager.currentCallId) {
                console.warn('[CallManager] Already in call, rejecting incoming');
                window.wsClient?.send({ type: 'call_reject', target: from, call_id: call_id });
                return;
            }
            window.CallManager.showIncomingCallModal(call_id, from, sdp, from_name, candidates || [], video || false);
        }
        else if (type === 'call_answer') {
            window.CallManager.handleRemoteAnswer(sdp);
        }
        else if (type === 'call_ice') {
            window.CallManager.handleRemoteIce(candidate);
        }
        else if (type === 'call_hangup') {
            window.CallManager.endCall();
        }
        else if (type === 'call_reject') {
            window.CallManager.endCall();
        }
    };

})();