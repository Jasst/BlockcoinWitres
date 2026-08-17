(function() {
  'use strict';

  const NotificationManager = {
    audioCtx: null,
    userInteracted: false,
    originalTitle: document.title,
    blinkInterval: null,
    initialized: false,
    lastNotificationTime: new Map(),
    activeChatId: null,
    processedMessageIds: new Set(),

    config: {
      soundEnabled: true,
      notificationEnabled: true,
      blinkEnabled: true,
      throttleMs: 2000,
      maxTracked: 100,
      notificationTimeout: 8000,
      toastDuration: 5000
    },

    async init() {
      if (this.initialized) return;
      this.initialized = true;

      const markInteraction = () => {
        if (this.userInteracted) return;
        this.userInteracted = true;
        this._ensureAudioContext();
        document.removeEventListener('click', markInteraction);
        document.removeEventListener('touchstart', markInteraction);
        document.removeEventListener('pointerdown', markInteraction);
        document.removeEventListener('keydown', markInteraction);
      };
      document.addEventListener('click', markInteraction, { passive: true });
      document.addEventListener('touchstart', markInteraction, { passive: true });
      document.addEventListener('pointerdown', markInteraction, { passive: true });
      document.addEventListener('keydown', markInteraction, { passive: true });
    },

    async requestNotificationPermission() {
      if (!('Notification' in window)) return false;
      if (Notification.permission === 'granted') {
        if (window.initPushNotifications) window.initPushNotifications();
        return true;
      }
      if (Notification.permission === 'denied') return false;
      try {
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
          if (window.initPushNotifications) window.initPushNotifications();
          return true;
        }
      } catch (e) {
        console.debug('Notification request failed', e);
      }
      return false;
    },

    _ensureAudioContext() {
      if (!this.audioCtx) {
        try {
          this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
          console.debug('Web Audio not available');
          return false;
        }
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume().catch(() => {});
      }
      return this.audioCtx.state !== 'closed';
    },

    setActiveChat(chatId) {
      this.activeChatId = chatId ? this._normalizeChatId(chatId) : null;
      this.stopBlink();

      // ✅ Сообщаем Service Worker
      this._syncActiveChatToSW();
    },

    // ✅ НОВОЕ: Централизованная синхронизация с SW
    _syncActiveChatToSW() {
      if (navigator.serviceWorker?.controller) {
        if (this.activeChatId) {
          navigator.serviceWorker.controller.postMessage({
            type: 'SET_ACTIVE_CHAT',
            chatId: this.activeChatId
          });
        } else {
          navigator.serviceWorker.controller.postMessage({
            type: 'CLEAR_ACTIVE_CHAT'
          });
        }
      }
    },

    isActiveChat(chatId) {
      if (document.visibilityState !== 'visible') return false;
      const current = this.activeChatId;
      const target = this._normalizeChatId(chatId);
      return !!(current && target && current === target);
    },

    playSound(type = 'received') {
      if (!this.userInteracted) return;
      if (!this.config.soundEnabled) return;
      if (!this._ensureAudioContext()) return;

      try {
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        const profiles = {
          received: { freq: 1000, duration: 0.15, gain: 0.08 },
          sent:     { freq: 600,  duration: 0.1,  gain: 0.05 },
          default:  { freq: 800,  duration: 0.2,  gain: 0.1  }
        };
        const { freq, duration, gain: g } = profiles[type] || profiles.default;

        osc.frequency.value = freq;
        osc.type = 'sine';
        gain.gain.setValueAtTime(g, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);
        osc.start();
        osc.stop(this.audioCtx.currentTime + duration);
      } catch (e) {
        console.debug('Sound play failed:', e);
      }
    },

    _normalizeChatId(chatId) {
      if (!chatId) return '';
      return chatId.startsWith('group:') ? chatId.slice(6) : chatId;
    },

    handleIncomingMessage(msg) {
      if (!msg?.sender || !msg?.chatId) return;

      if (msg.messageId) {
        if (this.processedMessageIds.has(msg.messageId)) return;
        this.processedMessageIds.add(msg.messageId);
        if (this.processedMessageIds.size > this.config.maxTracked) {
          const iter = this.processedMessageIds.values();
          const deleteCount = Math.floor(this.config.maxTracked / 2);
          for (let i = 0; i < deleteCount; i++) {
            this.processedMessageIds.delete(iter.next().value);
          }
        }
      }

      if (this.isActiveChat(msg.chatId)) return;

      const now = Date.now();
      const lastTime = this.lastNotificationTime.get(msg.chatId) || 0;
      const shouldThrottle = (now - lastTime) < this.config.throttleMs;
      if (!shouldThrottle) {
        this.lastNotificationTime.set(msg.chatId, now);
        this.playSound('received');
        this.showToastForMessage(msg);
      }

      if (document.visibilityState !== 'visible') {
        if (this.config.blinkEnabled) {
          this.startBlink(msg.preview || 'Новое сообщение');
        }
      }
    },

    showToastForMessage(msg) {
      const senderName = (msg.sender?.slice(0, 12) || '?') + '…';
      const preview = msg.content || msg.preview || 'Новое сообщение';
      const chatId = msg.chatId;
      const isGroup = msg.isGroup;

      document.querySelectorAll('.in-app-notification').forEach(el => el.remove());

      const toast = document.createElement('div');
      toast.className = 'in-app-notification';
      toast.innerHTML = `
        <div class="in-app-notification-content" style="cursor:pointer;">
          <div class="in-app-avatar">${(senderName[0] || '?').toUpperCase()}</div>
          <div class="in-app-body">
            <div class="in-app-sender">${this._escapeHtml(senderName)}</div>
            <div class="in-app-preview">${this._escapeHtml(preview)}</div>
          </div>
          <button class="in-app-close" aria-label="Close">&times;</button>
        </div>
      `;

      toast.querySelector('.in-app-notification-content').onclick = () => {
        if (typeof window.selectConversation === 'function') {
          window.selectConversation(chatId, senderName, isGroup);
        } else {
          const params = new URLSearchParams({ start_with: chatId, name: senderName || 'Contact' });
          window.location.href = '/chat?' + params.toString();
        }
        toast.remove();
      };

      toast.querySelector('.in-app-close').onclick = (e) => {
        e.stopPropagation();
        toast.remove();
      };

      document.body.appendChild(toast);
      setTimeout(() => { if (toast.parentNode) toast.remove(); }, this.config.toastDuration);
    },

    startBlink(text) {
      if (this.blinkInterval) clearInterval(this.blinkInterval);
      let toggle = true;
      const orig = this.originalTitle;
      this.blinkInterval = setInterval(() => {
        if (document.visibilityState === 'visible') { this.stopBlink(); return; }
        document.title = toggle ? `🔔 ${text}` : orig;
        toggle = !toggle;
      }, 1000);
    },

    stopBlink() {
      if (this.blinkInterval) { clearInterval(this.blinkInterval); this.blinkInterval = null; }
      document.title = this.originalTitle;
    },

    showToast(message, type = 'info', duration = 4000) {
      document.querySelectorAll('.system-toast').forEach(n => n.remove());
      const toast = document.createElement('div');
      toast.className = `system-toast ${type}`;
      toast.setAttribute('role', 'alert');
      toast.innerHTML = `<span class="icon">${this._getToastIcon(type)}</span>
                         <span>${this._escapeHtml(message)}</span>
                         <button class="close">&times;</button>`;
      const close = () => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 200); };
      toast.querySelector('.close').onclick = close;
      document.body.appendChild(toast);
      setTimeout(close, duration);
    },

    _getToastIcon(type) {
      const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
      return icons[type] || icons.info;
    },

    _escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str || '';
      return div.innerHTML;
    },

    configure(cfg) { this.config = { ...this.config, ...cfg }; },

    destroy() {
      this.stopBlink();
      if (this.audioCtx?.state !== 'closed') this.audioCtx?.close().catch(() => {});
      this.processedMessageIds.clear();
      this.initialized = false;
    }
  };

  window.NotificationManager = NotificationManager;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => NotificationManager.init());
  } else {
    NotificationManager.init();
  }

  // ✅ При уходе со страницы — сбрасываем активный чат в SW
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      NotificationManager.stopBlink();
      document.title = NotificationManager.originalTitle;
      // ✅ При возвращении на страницу — обновляем activeChatId в SW
      NotificationManager._syncActiveChatToSW();
    } else {
      // ✅ При уходе страницы в фон — сбрасываем activeChatId в SW
      // чтобы push-уведомления показывались для ВСЕХ чатов
      if (navigator.serviceWorker?.controller) {
        navigator.serviceWorker.controller.postMessage({
          type: 'CLEAR_ACTIVE_CHAT'
        });
      }
    }
  });

  window.addEventListener('beforeunload', () => {
    // ✅ При закрытии/перезагрузке — сбрасываем
    if (navigator.serviceWorker?.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'CLEAR_ACTIVE_CHAT'
      });
    }
    NotificationManager.destroy();
  });
})();