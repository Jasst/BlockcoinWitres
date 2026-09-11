// Audio_client.js — расширение DarkCrypto (подключать после crypto-client.js)
(function() {
    if (typeof window.DarkCrypto === 'undefined') {
        console.error('DarkCrypto not loaded yet');
        return;
    }

    // Шифрование файла
    DarkCrypto.encryptFile = async function(fileData, key, iv) {
        const cryptoKey = await crypto.subtle.importKey('raw', key, { name: 'AES-GCM' }, false, ['encrypt']);
        const encrypted = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, cryptoKey, fileData);
        return new Uint8Array(encrypted);
    };

    // Дешифрование файла
    DarkCrypto.decryptFile = async function(encryptedData, key, iv) {
        const cryptoKey = await crypto.subtle.importKey('raw', key, { name: 'AES-GCM' }, false, ['decrypt']);
        const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, cryptoKey, encryptedData);
        return decrypted;
    };

    // Генерация случайного ключа и IV для AES-GCM
    DarkCrypto.generateFileKeyAndIv = function() {
        const key = crypto.getRandomValues(new Uint8Array(32));
        const iv = crypto.getRandomValues(new Uint8Array(12));
        return { key, iv };
    };

    // ArrayBuffer → Base64
    DarkCrypto.arrayBufferToBase64 = function(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    };

    // Base64 → ArrayBuffer
    DarkCrypto.base64ToArrayBuffer = function(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
    };
})();