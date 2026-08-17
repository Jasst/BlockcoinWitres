/**
 * mnemonic-manager.js — Безопасное управление мнемоническими фразами
 * ✅ Защита от повторной загрузки + ФИКС копирования
 * ✅ Добавлено управление глобальным флагом window.modalOpen
 * ✅ Обработка закрытия модалки через крестик
 */

// === 🛡️ Защита от повторного объявления ===
if (typeof window.MnemonicManager !== 'undefined') {
  console.debug('ℹ️ MnemonicManager already loaded, skipping');
} else {
  (function() {
    const MnemonicManager = {
      _mnemonic: null,
      _clearTimer: null,
      _autoClearSeconds: 30,

      showInModal(mnemonic, modalId = 'mnemonicModal', displayId = 'modalMnemonic') {
    if (!mnemonic) return;
    this._mnemonic = mnemonic;
    const modal = document.getElementById(modalId);
    const display = document.getElementById(displayId);
    if (display) display.textContent = mnemonic;
    if (modal) {
        modal.classList.remove('hidden');
        modal.setAttribute('aria-hidden', 'false');
        // ✅ Устанавливаем глобальный флаг, чтобы Long Polling не обновлял UI
        window.modalOpen = true;
        // ✅ Локализуем только статичные data-i18n элементы внутри модалки,
        //    не трогая #clearCountdown (он управляется таймером)
        if (window.i18next) {
            modal.querySelectorAll('[data-i18n]').forEach(el => {
                if (el.id === 'clearCountdown') return;
                const key = el.getAttribute('data-i18n');
                const isInputLike = el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';
                if (isInputLike) {
                    el.placeholder = i18next.t(key);
                } else {
                    el.innerHTML = i18next.t(key);
                }
            });
            modal.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
                el.placeholder = i18next.t(el.getAttribute('data-i18n-placeholder'));
            });
        }
    }
    this._startAutoClear(modalId);
    this._preventEscClose(modalId, true);

    // ✅ Добавляем обработчик для крестика, чтобы корректно закрыть модалку через hide()
    const closeBtn = modal?.querySelector('.modal-close');
    if (closeBtn && !closeBtn._mnemonicHandler) {
        closeBtn._mnemonicHandler = () => this.hide(modalId);
        closeBtn.addEventListener('click', closeBtn._mnemonicHandler);
    }
},

      hide(modalId = 'mnemonicModal') {
        this._stopAutoClear();
        this._wipe();
        const modal = document.getElementById(modalId);
        if (modal) {
          modal.classList.add('hidden');
          modal.setAttribute('aria-hidden', 'true');
          this._preventEscClose(modalId, false);
          // ✅ Сбрасываем глобальный флаг
          window.modalOpen = false;
        }
        window.NotificationManager?.showToast('🧹 Mnemonic cleared from memory', 'success');
      },

      // ✅ ИСПРАВЛЕНО: надёжное копирование с фолбэком
      async copy(onSuccess, onError) {
        try {
          if (!this._mnemonic) {
            onError?.('Mnemonic not available');
            return false;
          }

          if (typeof Utils !== 'undefined' && typeof Utils.copyToClipboard === 'function') {
            return await Utils.copyToClipboard(this._mnemonic, onSuccess, onError);
          }

          console.log('📋 Using fallback clipboard method');
          const text = this._mnemonic;

          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            onSuccess?.();
            return true;
          }

          const textarea = document.createElement('textarea');
          textarea.value = text;
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          textarea.style.left = '-9999px';
          document.body.appendChild(textarea);
          textarea.select();
          const success = document.execCommand('copy');
          document.body.removeChild(textarea);

          if (success) {
            onSuccess?.();
            return true;
          } else {
            throw new Error('execCommand failed');
          }
        } catch (e) {
          console.error('❌ Copy failed:', e);
          onError?.(e.message || 'Copy failed');
          window.NotificationManager?.showToast('Copy failed: ' + e.message, 'error');
          return false;
        }
      },

      download(filename = null) {
        if (!this._mnemonic) return false;
        try {
          const blob = new Blob([this._mnemonic], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename || `mnemonic-${Date.now()}.txt`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          setTimeout(() => URL.revokeObjectURL(url), 100);
          return true;
        } catch (e) {
          console.error('Download error:', e);
          window.NotificationManager?.showToast('Download failed', 'error');
          return false;
        }
      },

      _startAutoClear(modalId) {
        this._stopAutoClear();
        let seconds = this._autoClearSeconds;
        const countdown = document.getElementById('clearCountdown');
        this._clearTimer = setInterval(() => {
          seconds--;
          if (countdown) countdown.textContent = seconds;
          if (seconds <= 0) this.hide(modalId);
        }, 1000);
      },

      _stopAutoClear() {
        if (this._clearTimer) { clearInterval(this._clearTimer); this._clearTimer = null; }
      },

      _wipe() {
        this._mnemonic = null;
      },

      _preventEscClose(modalId, prevent) {
        const modal = document.getElementById(modalId);
        if (!modal) return;
        if (prevent) {
          const handler = (e) => { if (e.key === 'Escape') e.preventDefault(); };
          modal._escHandler = handler;
          document.addEventListener('keydown', handler, { passive: false });
        } else if (modal._escHandler) {
          document.removeEventListener('keydown', modal._escHandler);
          delete modal._escHandler;
        }
      }
    };

    window.MnemonicManager = MnemonicManager;
    console.log('✅ MnemonicManager loaded');
  })();
}