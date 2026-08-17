/**
 * common.js — Shared utilities for Dark Messenger
 * Fully internationalized (i18n)
 * QR Scanner, Utils, DOM helpers, Security, Forms, Modals
 */
(function(global) {
  if (global.DarkMsgCommonLoaded) {
    console.debug('ℹ️ common.js already loaded, skipping');
    return;
  }
  global.DarkMsgCommonLoaded = true;
  window.modalOpen = false;

  // Helper for safe i18n (fallback to key if i18next not ready)
  function t(key, opts) {
    if (typeof i18next !== 'undefined' && i18next.t) {
      return i18next.t(key, opts);
    }
    // Fallback English for keys used in this file
    const fallbacks = {
      'just_now': 'just now',
      'minutes_ago': '{count}m ago',
      'address_required': 'Address is required',
      'must_be_64_hex': 'Must be 64 hex characters',
      'name_required': 'Name is required',
      'min_characters': 'Min {count} characters',
      'max_characters': 'Max {count} characters',
      'only_letters_numbers': 'Only letters, numbers, spaces, - _ allowed',
      'nothing_to_copy': 'Nothing to copy',
      'copy_failed': 'Copy failed',
      'processing': 'Processing…',
      'submit': 'Submit',
      'request_failed': 'Request failed',
      'camera_not_supported': 'Camera not supported in this browser',
      'camera_access_denied': 'Camera access denied',
      'no_camera_found': 'No camera found',
      'camera_busy': 'Camera is busy',
      'camera_settings_not_supported': 'Camera settings not supported',
      'camera_not_responding': 'Camera not responding',
      'address_scanned': '✓ Address scanned',
      'not_valid_address': '⚠️ Not a valid address format',
      'scanned_adding_contact': '✓ Scanned! Adding contact...',
      'scan_complete_manual_add': 'Scan complete. Please click "Add" manually.',
      'address_scanned_start_chat': '✓ Address scanned! Click "Start Chat" to begin',
      'cancel': 'Cancel',
      'confirm': 'Confirm',
      'ok': 'OK'
    };
    let result = fallbacks[key];
    if (result && opts) {
      if (opts.count !== undefined) result = result.replace('{count}', opts.count);
      if (opts.size) result = result.replace('{size}', opts.size);
    }
    return result || key;
  }

// =============================================================================
// === 🛡️ Security Module ===
// =============================================================================
const Security = {
  wipeSensitive(data) {
    if (!data) return null;
    if (typeof data === 'string') {
      data = data.split('').map(() => '\0').join('');
    }
    return null;
  },

  generateSafeId(length = 16) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    if (window.crypto?.getRandomValues) {
      const bytes = new Uint8Array(length);
      crypto.getRandomValues(bytes);
      for (let i = 0; i < length; i++) {
        result += chars[bytes[i] % chars.length];
      }
    } else {
      for (let i = 0; i < length; i++) {
        result += chars[Math.floor(Math.random() * chars.length)];
      }
    }
    return result;
  },

  isValidAddress(str) {
    return typeof str === 'string' && str.length === 64 && /^[a-fA-F0-9]{64}$/.test(str);
  },

  maskAddress(addr, visibleChars = 8) {
    if (!addr || addr.length <= visibleChars * 2) return addr;
    return addr.slice(0, visibleChars) + '…' + addr.slice(-visibleChars);
  }
};

// =============================================================================
// === 📋 Utilities ===
// =============================================================================
const Utils = {
  escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },

  escapeAttr(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;');
  },

  async copyToClipboard(text, onSuccess, onError) {
    if (!text) { onError?.(t('nothing_to_copy')); return false; }
    try {
      await navigator.clipboard.writeText(text);
      onSuccess?.();
      return true;
    } catch (e) {
      try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        onSuccess?.();
        return true;
      } catch (fallbackErr) {
        onError?.(fallbackErr.message || t('copy_failed'));
        return false;
      }
    }
  },

  formatTimestamp(ts) {
    if (!ts) return '';
    const date = new Date(ts * 1000);
    const now = new Date();
    const diff = now - date;
    if (diff < 60000) return t('just_now');
    if (diff < 3600000) return t('minutes_ago', { count: Math.floor(diff / 60000) });
    if (diff < 86400000) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return date.toLocaleDateString();
  },

  parseQRData(data) {
    if (!data) return null;
    if (Security.isValidAddress(data)) return data.toLowerCase();
    if (data.toLowerCase().startsWith('darkmsg:')) {
      const match = data.match(/darkmsg:([a-fA-F0-9]{64})/i);
      if (match?.[1]) return match[1].toLowerCase();
    }
    if (data.toLowerCase().startsWith('bitcoin:')) {
      const match = data.match(/bitcoin:([a-fA-F0-9]{64})/i);
      if (match?.[1]) return match[1].toLowerCase();
    }
    return null;
  },

  debounce(func, wait = 300) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => { clearTimeout(timeout); func(...args); };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }
};

// =============================================================================
// === 🎨 DOM Helpers ===
// =============================================================================
const DOM = {
  getById(id) { return document.getElementById(id); },
  query(selector, root = document) { return root.querySelector(selector); },
  queryAll(selector, root = document) { return Array.from(root.querySelectorAll(selector)); },

  on(selector, event, handler, options) {
    const el = typeof selector === 'string' ? document.querySelector(selector) : selector;
    if (el) el.addEventListener(event, handler, options);
    return () => el?.removeEventListener(event, handler, options);
  },

  delegate(parent, selector, event, handler) {
    const el = typeof parent === 'string' ? document.querySelector(parent) : parent;
    if (!el) return () => {};
    const listener = (e) => {
      const target = e.target.closest(selector);
      if (target && el.contains(target)) handler.call(target, e, target);
    };
    el.addEventListener(event, listener);
    return () => el.removeEventListener(event, listener);
  },

  toggleClass(el, className, force) {
    if (!el?.classList) return;
    el.classList.toggle(className, typeof force === 'boolean' ? force : undefined);
  },

  show(el) { el?.classList?.remove('hidden'); },
  hide(el) { el?.classList?.add('hidden'); },

  createElement(tag, { className, id, attributes = {}, text, html, children = [] } = {}) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (id) el.id = id;
    Object.entries(attributes).forEach(([k, v]) => el.setAttribute(k, v));
    if (text !== undefined) el.textContent = text;
    if (html) el.innerHTML = html;
    children.forEach(child => el.appendChild(child));
    return el;
  }
};

// =============================================================================
// === 🔲 Modal Manager ===
// =============================================================================
const ModalManager = {
  modals: new Map(),

  open(id, options = {}) {
    window.modalOpen = true;
    const modal = DOM.getById(id);
    if (!modal) return false;
    if (options.closeOthers) {
      this.modals.forEach((_, key) => { if (key !== id) this.close(key); });
    }
    DOM.show(modal);
    modal.setAttribute('aria-hidden', 'false');
    setTimeout(() => {
      const focusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
      focusable?.focus();
    }, 100);
    const onEsc = (e) => {
      if (e.key === 'Escape' && !options.preventEscClose) {
        e.preventDefault();
        this.close(id);
      }
    };
    document.addEventListener('keydown', onEsc);
    this.modals.set(id, { element: modal, onEsc });
    if (options.closeOnOverlay) {
      const overlayClick = (e) => { if (e.target === modal) this.close(id); };
      modal.addEventListener('click', overlayClick);
      this.modals.get(id).overlayClick = overlayClick;
    }
    options.onOpen?.(modal);
    return true;
  },

  close(id) {
    window.modalOpen = false;
    const modal = DOM.getById(id);
    if (!modal) return false;
    const data = this.modals.get(id);
    if (data?.onEsc) document.removeEventListener('keydown', data.onEsc);
    if (data?.overlayClick) modal.removeEventListener('click', data.overlayClick);
    DOM.hide(modal);
    modal.setAttribute('aria-hidden', 'true');
    this.modals.delete(id);
    data?.onClose?.(modal);
    return true;
  },

  initTriggers() {
    document.addEventListener('click', (e) => {
      const trigger = e.target.closest('[data-modal-trigger]');
      if (!trigger) return;
      e.preventDefault();
      const modalId = trigger.dataset.modalTrigger;
      const options = {
        closeOthers: trigger.dataset.closeOthers !== 'false',
        closeOnOverlay: trigger.dataset.closeOnOverlay !== 'false',
        preventEscClose: trigger.dataset.preventEscClose === 'true'
      };
      this.open(modalId, options);
    });
  }
};

// =============================================================================
// === 📝 Form Helpers ===
// =============================================================================
const FormHelpers = {
  validateAddress(value) {
    if (!value) return { valid: false, error: t('address_required') };
    if (!Security.isValidAddress(value)) {
      return { valid: false, error: t('must_be_64_hex') };
    }
    return { valid: true };
  },

  validateName(value, { minLength = 1, maxLength = 100, allowSpecial = false } = {}) {
    if (!value?.trim()) return { valid: false, error: t('name_required') };
    if (value.length < minLength) return { valid: false, error: t('min_characters', { count: minLength }) };
    if (value.length > maxLength) return { valid: false, error: t('max_characters', { count: maxLength }) };
    if (!allowSpecial && /[^a-zA-Z0-9\s\-_]/.test(value)) {
      return { valid: false, error: t('only_letters_numbers') };
    }
    return { valid: true };
  },

  attachValidation(input, validator, errorEl) {
    const showError = (msg) => {
      if (errorEl) { errorEl.textContent = msg; DOM.show(errorEl); }
      input.setAttribute('aria-invalid', 'true');
    };
    const clearError = () => {
      if (errorEl) { errorEl.textContent = ''; DOM.hide(errorEl); }
      input.removeAttribute('aria-invalid');
    };
    input.addEventListener('blur', () => {
      const result = validator(input.value);
      result.valid ? clearError() : showError(result.error);
    });
    input.addEventListener('input', () => {
      if (input.getAttribute('aria-invalid') === 'true') {
        const result = validator(input.value);
        if (result.valid) clearError();
      }
    });
    return () => { clearError(); };
  },

  async handleSubmit(formEl, submitBtn, handler) {
    if (!formEl || !handler) return;
    formEl.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (submitBtn) {
        submitBtn.disabled = true;
        const originalText = submitBtn.textContent;
        submitBtn.dataset.originalText = originalText;
        submitBtn.textContent = submitBtn.dataset.loadingText || t('processing');
      }
      try {
        await handler(new FormData(formEl));
      } catch (err) {
        console.error('Form submit error:', err);
        window.NotificationManager?.showToast(err.message || t('request_failed'), 'error');
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = submitBtn.dataset.originalText || t('submit');
        }
      }
    });
  }
};



// =============================================================================
// === 🖥️ Global Modal Dialogs (Confirm / Prompt) ===
// =============================================================================

window.showConfirmModal = function(title, message) {
    return new Promise((resolve) => {
        const modalId = 'global-confirm-modal-' + Date.now();
        const modalHtml = `
            <div id="${modalId}" class="modal-overlay hidden" role="dialog" aria-modal="true">
                <div class="modal" style="max-width: 400px;">
                    <header class="modal-header">
                        <h3>${Utils.escapeHtml(title)}</h3>
                        <button class="modal-close" data-close>&times;</button>
                    </header>
                    <div class="modal-body">
                        <p>${Utils.escapeHtml(message)}</p>
                    </div>
                    <footer class="modal-footer">
                        <button class="btn btn-ghost" data-action="cancel">${t('cancel')}</button>
                        <button class="btn btn-primary" data-action="confirm">${t('confirm')}</button>
                    </footer>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modalEl = document.getElementById(modalId);

        const close = (result) => {
            ModalManager.close(modalId);
            modalEl.remove();
            resolve(result);
        };

        ModalManager.open(modalId, { closeOnOverlay: false, preventEscClose: false });

        modalEl.querySelector('[data-close]')?.addEventListener('click', () => close(false));
        modalEl.querySelector('[data-action="cancel"]')?.addEventListener('click', () => close(false));
        modalEl.querySelector('[data-action="confirm"]')?.addEventListener('click', () => close(true));
    });
};

window.showPromptModal = function(title, placeholder, defaultValue = '') {
    return new Promise((resolve) => {
        const modalId = 'global-prompt-modal-' + Date.now();
        const modalHtml = `
            <div id="${modalId}" class="modal-overlay hidden" role="dialog" aria-modal="true">
                <div class="modal" style="max-width: 400px;">
                    <header class="modal-header">
                        <h3>${Utils.escapeHtml(title)}</h3>
                        <button class="modal-close" data-close>&times;</button>
                    </header>
                    <div class="modal-body">
                        <input type="text" id="prompt-input-${modalId}" class="input"
                               placeholder="${Utils.escapeHtml(placeholder)}"
                               value="${Utils.escapeHtml(defaultValue)}"
                               style="width: 100%;">
                    </div>
                    <footer class="modal-footer">
                        <button class="btn btn-ghost" data-action="cancel">${t('cancel')}</button>
                        <button class="btn btn-primary" data-action="ok">${t('ok')}</button>
                    </footer>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modalEl = document.getElementById(modalId);
        const input = modalEl.querySelector(`#prompt-input-${modalId}`);

        const close = (result) => {
            ModalManager.close(modalId);
            modalEl.remove();
            resolve(result);
        };

        ModalManager.open(modalId, { closeOnOverlay: false, preventEscClose: false });
        input?.focus();

        const handleOk = () => close(input?.value?.trim() || null);
        modalEl.querySelector('[data-close]')?.addEventListener('click', () => close(null));
        modalEl.querySelector('[data-action="cancel"]')?.addEventListener('click', () => close(null));
        modalEl.querySelector('[data-action="ok"]')?.addEventListener('click', handleOk);
        input?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleOk();
        });
    });
};

// =============================================================================
// === 📱 Mobile Sidebar Helpers ===
// =============================================================================
if (typeof window.toggleSidebar !== 'function') {
  window.toggleSidebar = function() {
    const sidebar = DOM.getById('sidebar');
    const overlay = DOM.getById('sidebarOverlay');
    DOM.toggleClass(sidebar, 'open');
    const isOpen = sidebar?.classList.contains('open');
    DOM.toggleClass(overlay, 'active', isOpen);
    document.body.style.overflow = isOpen ? 'hidden' : '';
  };
}
if (typeof window.closeSidebar !== 'function') {
  window.closeSidebar = function() {
    const sidebar = DOM.getById('sidebar');
    const overlay = DOM.getById('sidebarOverlay');
    sidebar?.classList.remove('open');
    overlay?.classList.remove('active');
    document.body.style.overflow = '';
  };
}

// =============================================================================
// === 🌐 Global Exports ===
// =============================================================================
window.Utils = Utils;
window.DOM = DOM;
window.Security = Security;
window.ModalManager = ModalManager;
window.FormHelpers = FormHelpers;




// =============================================================================
// === 🧹 Cleanup ===
// =============================================================================

document.addEventListener('DOMContentLoaded', () => { ModalManager.initTriggers(); });

})(typeof window !== 'undefined' ? window : this);