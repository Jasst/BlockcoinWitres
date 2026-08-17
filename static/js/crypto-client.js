/**
 * crypto-client.js — End-to-end шифрование на стороне браузера
 * Версия: 3.3 (исправленная)
 */

class DarkCrypto {
  // =========================================================================
  // 1. ГЕНЕРАЦИЯ МНЕМОНИКИ (BIP39) — использует внешний wordlist
  // =========================================================================
  static async generateMnemonic() {
    const wordlist = window.BIP39_WORDLIST;
    if (!wordlist || !Array.isArray(wordlist) || wordlist.length !== 2048) {
      throw new Error('BIP39 wordlist not loaded or invalid (expected 2048 words)');
    }

    const entropy = crypto.getRandomValues(new Uint8Array(32));
    const hash = await crypto.subtle.digest('SHA-256', entropy);
    const checksumBits = 8;
    const checksumByte = new Uint8Array(hash)[0];
    const checksum = checksumByte >> (8 - checksumBits);
    const fullBits = [];
    for (let i = 0; i < entropy.length; i++) {
      for (let b = 7; b >= 0; b--) {
        fullBits.push((entropy[i] >> b) & 1);
      }
    }
    for (let b = checksumBits - 1; b >= 0; b--) {
      fullBits.push((checksum >> b) & 1);
    }
    const words = [];
    for (let i = 0; i < 24; i++) {
      let index = 0;
      for (let j = 0; j < 11; j++) {
        index = (index << 1) | fullBits[i * 11 + j];
      }
      words.push(wordlist[index]);
    }
    return words.join(' ');
  }

  // =========================================================================
  // 2. ДЕРИВАЦИЯ КЛЮЧЕЙ ИЗ МНЕМОНИКИ
  // =========================================================================
  static async deriveKeyPair(mnemonic) {
    const seed = await this._mnemonicToSeed(mnemonic);
    const rawPrivate = new Uint8Array(seed.slice(0, 32));
    const d = this._normalizePrivateKey(rawPrivate);
    const point = this._derivePubPoint(d);

    const jwkSign = {
      kty: 'EC', crv: 'P-256',
      d: this._bytesToBase64Url(d),
      x: this._bytesToBase64Url(point.x),
      y: this._bytesToBase64Url(point.y),
      ext: true,
    };
    const signPrivateKey = await crypto.subtle.importKey(
      'jwk', jwkSign, { name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign']
    );
    const ecdhPrivateKey = await crypto.subtle.importKey(
      'jwk', jwkSign, { name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits']
    );
    const jwk = await crypto.subtle.exportKey('jwk', signPrivateKey);
    const xBytes = this._base64UrlToBytes(jwk.x);
    const yBytes = this._base64UrlToBytes(jwk.y);
    const prefix = (yBytes[31] % 2 === 0) ? 0x02 : 0x03;
    const compressed = new Uint8Array(33);
    compressed[0] = prefix;
    compressed.set(xBytes, 1);
    const hash = await crypto.subtle.digest('SHA-256', compressed);
    const address = Array.from(new Uint8Array(hash))
      .map(b => b.toString(16).padStart(2, '0')).join('');
    return { signPrivateKey, ecdhPrivateKey, compressedPubKey: compressed, address };
  }

  // =========================================================================
  // 3. ДЕКОМПРЕССИЯ ПУБЛИЧНОГО КЛЮЧА
  // =========================================================================
  static decompressPublicKey(compressedKey) {
    if (compressedKey.length !== 33 || (compressedKey[0] !== 0x02 && compressedKey[0] !== 0x03)) {
      throw new Error('Invalid compressed key');
    }
    const p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFn;
    const a = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFCn;
    const b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604Bn;
    const x = BigInt('0x' + Array.from(compressedKey.slice(1)).map(b => b.toString(16).padStart(2,'0')).join(''));
    const rhs = (x * x * x + a * x + b) % p;

    const modPow = (base, exp) => {
      let res = 1n;
      while (exp > 0n) {
        if (exp & 1n) res = (res * base) % p;
        base = (base * base) % p;
        exp >>= 1n;
      }
      return res;
    };

    let y = modPow(rhs, (p + 1n) / 4n);
    if ((y & 1n) !== (compressedKey[0] === 0x03 ? 1n : 0n)) {
      y = p - y;
    }
    const xBytes = this._to32Bytes(x);
    const yBytes = this._to32Bytes(y);
    const uncompressed = new Uint8Array(65);
    uncompressed[0] = 0x04;
    uncompressed.set(xBytes, 1);
    uncompressed.set(yBytes, 33);
    return uncompressed;
  }

  // =========================================================================
  // 4. ECDH ОБЩИЙ СЕКРЕТ
  // =========================================================================
  static async getSharedSecret(myEcdhPrivateKey, theirPubKeyBytes) {
    let pubKey = theirPubKeyBytes;
    if (pubKey.length === 33 && (pubKey[0] === 0x02 || pubKey[0] === 0x03)) {
      pubKey = this.decompressPublicKey(pubKey);
    }
    const pubKeyObj = await crypto.subtle.importKey(
      'raw', pubKey, { name: 'ECDH', namedCurve: 'P-256' }, false, []
    );
    const shared = await crypto.subtle.deriveBits(
      { name: 'ECDH', public: pubKeyObj }, myEcdhPrivateKey, 256
    );
    return shared;
  }

  // =========================================================================
  // 5. AES-GCM ШИФРОВАНИЕ / ДЕШИФРОВАНИЕ
  // =========================================================================
  static async encryptAES(sharedSecret, plaintext) {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await crypto.subtle.importKey(
      'raw', sharedSecret, { name: 'AES-GCM' }, false, ['encrypt']
    );
    const encoded = new TextEncoder().encode(plaintext);
    const ciphertext = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv }, key, encoded
    );
    return { ciphertext, iv };
  }

  static async decryptAES(sharedSecret, ciphertext, iv) {
    const key = await crypto.subtle.importKey(
      'raw', sharedSecret, { name: 'AES-GCM' }, false, ['decrypt']
    );
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv }, key, ciphertext
    );
    return new TextDecoder().decode(decrypted);
  }

  // =========================================================================
  // 6. ШИФРОВАНИЕ / ДЕШИФРОВАНИЕ СООБЩЕНИЙ
  // =========================================================================
  static async encryptMessage(myEcdhPrivateKey, myCompressedPubKey, recipientPubKey, plaintext) {
    const shared = await this.getSharedSecret(myEcdhPrivateKey, recipientPubKey);
    const { ciphertext, iv } = await this.encryptAES(shared, plaintext);
    return {
      ciphertext: this._arrayBufferToBase64(ciphertext),
      iv: this._toBase64(iv),
      myPubKey: this._toBase64(myCompressedPubKey)
    };
  }

  static async decryptMessage(myEcdhPrivateKey, senderCompressedPubKey, ivBase64, ciphertextBase64) {
    const iv = this._fromBase64(ivBase64);
    const ciphertext = this._base64ToArrayBuffer(ciphertextBase64);
    const shared = await this.getSharedSecret(myEcdhPrivateKey, senderCompressedPubKey);
    return await this.decryptAES(shared, ciphertext, iv);
  }

  // =========================================================================
  // 7. ПОДПИСЬ ДАННЫХ
  // =========================================================================
  static async signData(privateKey, dataString) {
    const signature = await crypto.subtle.sign(
      { name: 'ECDSA', hash: 'SHA-256' },
      privateKey,
      new TextEncoder().encode(dataString)
    );
    return new Uint8Array(signature);
  }

  // =========================================================================
  // 8. ВЕРИФИКАЦИЯ ПОДПИСИ
  // =========================================================================
  static async verifySignature(publicKeyBytes, signature, dataString) {
    const pubKeyObj = await crypto.subtle.importKey(
      'raw', publicKeyBytes, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']
    );
    return await crypto.subtle.verify(
      { name: 'ECDSA', hash: 'SHA-256' },
      pubKeyObj,
      signature,
      new TextEncoder().encode(dataString)
    );
  }

  // =========================================================================
  // 9. ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
  // =========================================================================
  static async _mnemonicToSeed(mnemonic) {
    const keyMaterial = await crypto.subtle.importKey(
      'raw', new TextEncoder().encode(mnemonic),
      'PBKDF2', false, ['deriveBits']
    );
    return crypto.subtle.deriveBits(
      { name: 'PBKDF2', salt: new TextEncoder().encode('mnemonic'),
        iterations: 2048, hash: 'SHA-512' },
      keyMaterial, 512
    );
  }

  static _normalizePrivateKey(rawBytes) {
    const n = BigInt("0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551");
    let scalar = 0n;
    for (let i = 0; i < rawBytes.length; i++) {
      scalar = (scalar << 8n) | BigInt(rawBytes[i]);
    }
    scalar = (scalar % (n - 1n)) + 1n;
    const hex = scalar.toString(16).padStart(64, '0');
    const bytes = new Uint8Array(32);
    for (let i = 0; i < 32; i++) {
      bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
    }
    return bytes;
  }

  static _derivePubPoint(privateScalar) {
    const p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFn;
    const a = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFCn;
    const Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296n;
    const Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5n;
    const n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551n;

    let d = 0n;
    for (let i = 0; i < privateScalar.length; i++) {
      d = (d << 8n) | BigInt(privateScalar[i]);
    }
    d = d % n;
    if (d === 0n) {
      return { x: this._to32Bytes(Gx), y: this._to32Bytes(Gy) };
    }

    const mod = (x) => {
      let r = x % p;
      return r < 0n ? r + p : r;
    };
    const modAdd = (x, y) => mod((x + y) % p);
    const modSub = (x, y) => mod((x - y + p) % p);
    const modMul = (x, y) => mod((x * y) % p);

    const modInv = (x) => {
      if (x === 0n) throw new Error('Division by zero');
      let a = mod(x), prevA = 1n;
      let b = p, prevB = 0n;
      while (b !== 0n) {
        const q = a / b;
        [a, b] = [b, a - q * b];
        [prevA, prevB] = [prevB, prevA - q * prevB];
      }
      return mod(prevA);
    };

    const modDiv = (x, y) => modMul(x, modInv(y));

    const double = (P) => {
      if (P === null) return null;
      const [x, y] = [P.x, P.y];
      if (y === 0n) return null;
      const s = modDiv(modAdd(modMul(3n, modMul(x, x)), a), modMul(2n, y));
      const x2 = modSub(modMul(s, s), modMul(2n, x));
      const y2 = modSub(modMul(s, modSub(x, x2)), y);
      return { x: x2, y: y2 };
    };

    const add = (P, Q) => {
      if (P === null) return Q;
      if (Q === null) return P;
      if (P.x === Q.x) {
        if (P.y !== Q.y) return null;
        return double(P);
      }
      const s = modDiv(modSub(Q.y, P.y), modSub(Q.x, P.x));
      const x3 = modSub(modMul(s, s), modAdd(P.x, Q.x));
      const y3 = modSub(modMul(s, modSub(P.x, x3)), P.y);
      return { x: x3, y: y3 };
    };

    let Q = null;
    let R = { x: Gx, y: Gy };
    while (d > 0n) {
      if (d & 1n) {
        Q = add(Q, R);
      }
      R = double(R);
      d >>= 1n;
    }
    return { x: this._to32Bytes(Q.x), y: this._to32Bytes(Q.y) };
  }

  static _to32Bytes(value) {
    const hex = value.toString(16).padStart(64, '0');
    const bytes = new Uint8Array(32);
    for (let i = 0; i < 32; i++) {
      bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
    }
    return bytes;
  }

  static _bytesToBase64Url(bytes) {
    let binary = '';
    bytes.forEach(b => binary += String.fromCharCode(b));
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  static _base64UrlToBytes(base64url) {
    let b64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
    while (b64.length % 4) b64 += '=';
    const raw = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return bytes;
  }

  static _toBase64(arr) {
    return btoa(String.fromCharCode(...arr));
  }

  static _fromBase64(str) {
    return new Uint8Array(atob(str).split('').map(c => c.charCodeAt(0)));
  }

  static _arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    return this._toBase64(bytes);
  }

  static _base64ToArrayBuffer(base64) {
    const bytes = this._fromBase64(base64);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  }

  static _concat(a, b) {
    const c = new Uint8Array(a.length + b.length);
    c.set(a, 0);
    c.set(b, a.length);
    return c;
  }
}

// Экспортируем в глобальную область видимости
window.DarkCrypto = DarkCrypto;