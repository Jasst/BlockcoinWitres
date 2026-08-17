// static/js/qr-manager.js
(function(global) {
    if (global.QRManager) return;

    // === QRScanner (универсальный сканер) ===
    const QRScanner = {
        stream: null,
        animationFrame: null,
        active: false,
        _videoEl: null,
        _containerEl: null,
        _resultEl: null,
        config: {
            videoWidth: 1280,
            videoHeight: 720,
            scanSize: 280,
            inversionAttempts: "attemptBoth",
            scanInterval: 80,
            videoWaitTimeout: 5000,
            videoWaitInterval: 100
        },

        open(options) {
            if (this.active) return;
            const { videoEl, resultEl, containerEl, onScan, onClose } = options;
            if (!videoEl || !containerEl) {
                console.error('QRScanner: missing video or container element');
                return;
            }
            this.active = true;
            this._videoEl = videoEl;
            this._containerEl = containerEl;
            this._resultEl = resultEl || null;

            // Показываем контейнер
            DOM.show(containerEl);
            containerEl.classList.add('scanning');
            if (resultEl) {
                DOM.hide(resultEl);
                resultEl.textContent = '';
                resultEl.style.color = '';
            }

            if (!navigator.mediaDevices?.getUserMedia) {
                this._showError(t('camera_not_supported'), resultEl, onClose);
                return;
            }

            navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: this.config.videoWidth, min: 640 },
                    height: { ideal: this.config.videoHeight, min: 480 }
                },
                audio: false
            })
            .then(stream => {
                if (!this.active) { this._stopStream(stream); return; }
                this.stream = stream;
                videoEl.srcObject = stream;
                return videoEl.play().catch(err => { console.error('Video play failed:', err); throw err; });
            })
            .then(() => {
                let attempts = 0;
                const maxAttempts = this.config.videoWaitTimeout / this.config.videoWaitInterval;
                const checkVideo = () => {
                    if (!this.active) return;
                    if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
                        this._startScanning({ videoEl, resultEl, onScan, containerEl });
                    } else {
                        attempts++;
                        if (attempts >= maxAttempts) {
                            console.warn('QRScanner: video not ready after timeout');
                            this._showError(t('camera_not_responding'), resultEl, onClose);
                            this.close();
                            return;
                        }
                        setTimeout(checkVideo, this.config.videoWaitInterval);
                    }
                };
                checkVideo();
            })
            .catch(err => {
                console.error('QRScanner error:', err.name, err.message);
                let userMessage = t('camera_access_denied');
                if (err.name === 'NotFoundError') userMessage = t('no_camera_found');
                if (err.name === 'NotReadableError') userMessage = t('camera_busy');
                if (err.name === 'OverconstrainedError') userMessage = t('camera_settings_not_supported');
                this._showError(userMessage, resultEl, onClose);
            });
        },

        close() {
            this.active = false;
            if (this.animationFrame) {
                cancelAnimationFrame(this.animationFrame);
                this.animationFrame = null;
            }
            if (this.stream) {
                this._stopStream(this.stream);
                this.stream = null;
            }
            if (this._containerEl) {
                DOM.hide(this._containerEl);
                this._containerEl.classList.remove('scanning');
            }
            if (this._videoEl) {
                this._videoEl.pause();
                this._videoEl.srcObject = null;
            }
            if (this._resultEl) DOM.hide(this._resultEl);
            // Сбрасываем ссылки
            this._videoEl = null;
            this._containerEl = null;
            this._resultEl = null;
        },

        _stopStream(stream) {
            if (stream?.getTracks) {
                stream.getTracks().forEach(track => track.stop());
            }
        },

        _showError(message, resultEl, onClose) {
            console.warn('QRScanner error shown to user:', message);
            window.NotificationManager?.showToast(message, 'error');
            if (resultEl) {
                resultEl.textContent = '⚠️ ' + message;
                resultEl.style.color = 'var(--status-error)';
                DOM.show(resultEl);
            }
            setTimeout(() => {
                if (!this.active) this.close();
            }, 3000);
        },

        _startScanning({ videoEl, resultEl, onScan, containerEl }) {
            if (!this.active || !videoEl) return;
            let canvas = document.createElement('canvas');
            canvas.width = this.config.scanSize;
            canvas.height = this.config.scanSize;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            const { scanSize, inversionAttempts, scanInterval } = this.config;
            let lastScanTime = 0;

            const scanFrame = () => {
                if (!this.active) return;
                const now = performance.now();
                if (now - lastScanTime < scanInterval) {
                    this.animationFrame = requestAnimationFrame(scanFrame);
                    return;
                }
                lastScanTime = now;

                if (videoEl.readyState !== videoEl.HAVE_ENOUGH_DATA || !videoEl.videoWidth || !videoEl.videoHeight) {
                    this.animationFrame = requestAnimationFrame(scanFrame);
                    return;
                }

                try {
                    const videoW = videoEl.videoWidth;
                    const videoH = videoEl.videoHeight;
                    const size = Math.min(videoW, videoH) * 0.5;
                    const sx = (videoW - size) / 2;
                    const sy = (videoH - size) / 2;

                    canvas.width = scanSize;
                    canvas.height = scanSize;
                    ctx.drawImage(videoEl, sx, sy, size, size, 0, 0, scanSize, scanSize);

                    const imageData = ctx.getImageData(0, 0, scanSize, scanSize);
                    const code = jsQR(imageData.data, scanSize, scanSize, { inversionAttempts });

                    if (code?.data) {
                        const address = Utils.parseQRData(code.data);
                        if (address) {
                            onScan?.(address);
                            this.close();
                            if (resultEl) {
                                resultEl.textContent = t('address_scanned');
                                resultEl.style.color = 'var(--status-success)';
                                DOM.show(resultEl);
                            }
                            onScan?.(address);
                            window.NotificationManager?.showToast(t('address_scanned'), 'success');
                            return;
                        } else {
                            console.warn('⚠️ QR data not recognized as address:', code.data);
                            if (resultEl) {
                                resultEl.textContent = t('not_valid_address');
                                resultEl.style.color = 'var(--status-warning)';
                                DOM.show(resultEl);
                            }
                        }
                    }
                } catch (e) { console.debug('Scan frame error (non-critical):', e.message); }

                if (this.active) this.animationFrame = requestAnimationFrame(scanFrame);
            };
            this.animationFrame = requestAnimationFrame(scanFrame);
        }
    };

    // === Генерация QR ===
    function generateQRCode(element, text, options = {}) {
        if (!element) return;
        if (typeof QRCode === 'undefined') {
            console.warn('QRCode library not loaded');
            return;
        }
        // в функции generateQRCode
        const defaultOptions = {
    width: 300,
    height: 320,
    colorDark: '#000000',   // чёрные модули
    colorLight: '#ffffff',  // белый фон
    correctLevel: QRCode.CorrectLevel.H
};
        const merged = { ...defaultOptions, ...options };
        element.innerHTML = '';
        try {
            new QRCode(element, { text, ...merged });
        } catch(e) {
            element.innerHTML = '<span class="text-muted">QR error</span>';
        }
    }

    // === Предопределённые сценарии открытия сканера ===
    function openForChat() {
    // Сначала открываем модалку нового чата
    if (typeof window.openNewChatModal === 'function') {
        window.openNewChatModal(); // эта функция должна существовать (она есть в actions.js)
    } else {
        // fallback: показать модалку вручную
        const modal = document.getElementById('newChatModal');
        if (modal) modal.classList.remove('hidden');
    }

    // Даём время на рендеринг, затем запускаем сканер
    setTimeout(() => {
        const video = document.getElementById('qrVideoModal');
        const result = document.getElementById('scanResultModal');
        const container = document.getElementById('qrScannerContainerModal');
        if (!video || !container) {
            console.warn('QR elements for chat not found');
            return;
        }
        QRScanner.open({
            videoEl: video,
            resultEl: result,
            containerEl: container,
            onScan: (addr) => {
                const addressInput = document.getElementById('newChatAddress');
                if (addressInput) {
                    addressInput.value = addr;
                    if (result) {
                        result.textContent = '✓ Address scanned! Click "Start Chat" to begin';
                        result.style.color = 'var(--status-success)';
                        DOM.show(result);
                    }
                }
            }
        });
    }, 300); // небольшая задержка, чтобы модалка успела открыться
}

    function openForContacts() {
        const video = document.getElementById('qrVideo');
        const result = document.getElementById('scanResult');
        const container = document.getElementById('qrScannerContainer');
        if (!video || !container) {
            console.warn('QR elements for contacts not found');
            return;
        }
        QRScanner.open({
            videoEl: video,
            resultEl: result,
            containerEl: container,
            onScan: async (addr) => {
                const addressEl = document.getElementById('contactAddress');
                const nameEl = document.getElementById('contactName');
                if (addressEl) addressEl.value = addr;
                if (nameEl && !nameEl.value.trim()) {
                    nameEl.value = 'Contact_' + addr.slice(0, 8);
                }
                if (result) {
                    result.textContent = '✓ Scanned! Adding contact...';
                    result.style.color = 'var(--status-success)';
                    DOM.show(result);
                }
                setTimeout(async () => {
                    if (addressEl?.value && nameEl?.value && typeof window.addContact === 'function') {
                        try {
                            await window.addContact();
                        } catch (e) {
                            console.error('Auto-add failed:', e);
                            window.NotificationManager?.showToast('Scan complete. Please click "Add" manually.', 'warning');
                        }
                    }
                }, 400);
            }
        });
    }

    function openForWallet() {
        const video = document.getElementById('qrVideo');
        const result = document.getElementById('scanResult');
        const container = document.getElementById('qrScannerContainer');
        if (!video || !container) {
            console.warn('QR elements for wallet not found');
            return;
        }
        QRScanner.open({
            videoEl: video,
            resultEl: result,
            containerEl: container,
            onScan: (addr) => {
                const addressInput = document.getElementById('sendAddress');
                if (addressInput) {
                    addressInput.value = addr;
                    if (result) {
                        result.textContent = '✓ Address scanned! Ready to send.';
                        result.style.color = 'var(--status-success)';
                        DOM.show(result);
                    }
                }
            }
        });
    }

    // === Глобальный API ===
    global.QRManager = {
        generateQRCode,
        openForChat,
        openForContacts,
        openForWallet,
        closeScanner: () => QRScanner.close()
    };

})(window);