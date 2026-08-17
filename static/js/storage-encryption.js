// static/js/storage-encryption.js
//
// ИСПРАВЛЕНИЯ:
// 1. PBKDF2 вынесен в Web Worker — не блокирует UI на слабых устройствах
// 2. Добавлена поддержка onProgress-колбэка для отображения индикатора загрузки
// 3. Fallback на main thread если Workers недоступны (например, в некоторых iOS WebView)

(function() {
  if (window.StorageEncryption) return;
  window.StorageEncryption = {};

  // --- Утилиты ---

  function buf2hex(buffer) {
    return Array.from(new Uint8Array(buffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  function hex2buf(hex) {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) {
      bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
    }
    return bytes.buffer;
  }

  // --- PBKDF2 в main thread (fallback) ---

  async function pbkdf2MainThread(password, salt, iterations = 100000, length = 32) {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw', enc.encode(password), 'PBKDF2', false, ['deriveBits']
    );
    return await crypto.subtle.deriveBits(
      { name: 'PBKDF2', salt, iterations, hash: 'SHA-256' },
      keyMaterial, length * 8
    );
  }

  // --- PBKDF2 через Web Worker (не блокирует UI) ---
  //
  // Код воркера встраивается как Blob — не требует отдельного файла.
  // Это решает проблему блокировки main thread на 1–3 секунды при 100k итерациях
  // на слабых мобильных устройствах.

  const WORKER_CODE = `
    self.onmessage = async function(e) {
      const { password, saltHex, iterations, length } = e.data;
      try {
        const salt = new Uint8Array(saltHex.match(/../g).map(h => parseInt(h, 16)));
        const enc = new TextEncoder();
        const keyMaterial = await crypto.subtle.importKey(
          'raw', enc.encode(password), 'PBKDF2', false, ['deriveBits']
        );
        const bits = await crypto.subtle.deriveBits(
          { name: 'PBKDF2', salt, iterations, hash: 'SHA-256' },
          keyMaterial, length * 8
        );
        const hex = Array.from(new Uint8Array(bits))
          .map(b => b.toString(16).padStart(2, '0'))
          .join('');
        self.postMessage({ ok: true, hex });
      } catch (err) {
        self.postMessage({ ok: false, error: err.message });
      }
    };
  `;

  let _workerBlob = null;

  function createWorker() {
    try {
      if (!_workerBlob) {
        _workerBlob = URL.createObjectURL(
          new Blob([WORKER_CODE], { type: 'application/javascript' })
        );
      }
      return new Worker(_workerBlob);
    } catch (e) {
      console.debug('Web Worker unavailable, falling back to main thread', e);
      return null;
    }
  }

  function pbkdf2Worker(password, salt, iterations = 100000, length = 32) {
    return new Promise((resolve, reject) => {
      const worker = createWorker();
      if (!worker) {
        // Fallback: main thread
        pbkdf2MainThread(password, salt, iterations, length).then(resolve).catch(reject);
        return;
      }

      const saltHex = Array.from(new Uint8Array(salt))
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');

      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error('PBKDF2 Worker timeout'));
      }, 30000);

      worker.onmessage = (e) => {
        clearTimeout(timeout);
        worker.terminate();
        if (e.data.ok) {
          // Конвертируем hex обратно в ArrayBuffer
          const bytes = new Uint8Array(e.data.hex.match(/../g).map(h => parseInt(h, 16)));
          resolve(bytes.buffer);
        } else {
          reject(new Error(e.data.error || 'PBKDF2 failed'));
        }
      };

      worker.onerror = (err) => {
        clearTimeout(timeout);
        worker.terminate();
        // Fallback на main thread при ошибке воркера
        console.warn('Worker error, falling back to main thread:', err);
        pbkdf2MainThread(password, salt, iterations, length).then(resolve).catch(reject);
      };

      worker.postMessage({ password, saltHex, iterations, length });
    });
  }

  // --- Публичный API ---

  /**
   * Шифрует мнемонику паролем.
   * @param {string} mnemonic
   * @param {string} password
   * @param {function} [onProgress] — колбэк для индикатора загрузки (вызывается до и после)
   * @returns {Promise<string>} JSON-строка с зашифрованными данными
   */
  window.StorageEncryption.encryptMnemonic = async function(mnemonic, password, onProgress) {
    onProgress?.('Deriving key…');
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv   = crypto.getRandomValues(new Uint8Array(12));

    const keyBytes = await pbkdf2Worker(password, salt, 100000, 32);
    const key = await crypto.subtle.importKey('raw', keyBytes, 'AES-GCM', false, ['encrypt']);

    onProgress?.('Encrypting…');
    const enc = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      key,
      enc.encode(mnemonic)
    );

    onProgress?.(null); // сигнал завершения

    return JSON.stringify({
      v: 2, // версия схемы — для будущей совместимости
      salt: buf2hex(salt),
      iv:   buf2hex(iv),
      ciphertext: buf2hex(new Uint8Array(encrypted))
    });
  };

  /**
   * Расшифровывает мнемонику паролем.
   * @param {string} encryptedData — JSON из encryptMnemonic
   * @param {string} password
   * @param {function} [onProgress]
   * @returns {Promise<string|null>} мнемоника или null при неверном пароле
   */
  window.StorageEncryption.decryptMnemonic = async function(encryptedData, password, onProgress) {
    try {
      const { salt, iv, ciphertext } = JSON.parse(encryptedData);
      const saltBytes   = new Uint8Array(hex2buf(salt));
      const ivBytes     = new Uint8Array(hex2buf(iv));
      const cipherBytes = new Uint8Array(hex2buf(ciphertext));

      onProgress?.('Deriving key…');
      const keyBytes = await pbkdf2Worker(password, saltBytes, 100000, 32);
      const key = await crypto.subtle.importKey('raw', keyBytes, 'AES-GCM', false, ['decrypt']);

      onProgress?.('Decrypting…');
      const decrypted = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv: ivBytes },
        key,
        cipherBytes
      );

      onProgress?.(null);
      return new TextDecoder().decode(decrypted);
    } catch (e) {
      onProgress?.(null);
      console.warn('Decryption failed', e);
      return null;
    }
  };

  // Экспорт вспомогательных функций для использования в других модулях
  window.StorageEncryption._buf2hex = buf2hex;
  window.StorageEncryption._hex2buf = hex2buf;
})();