// wallet.js — полностью интернационализированная версия с поддержкой Web Worker, ETA и авто-майнинга
(function() {
  if (window._walletScriptLoaded) return;
  window._walletScriptLoaded = true;

  // Helper for i18n
  function t(key, opts) {
    if (typeof i18next !== 'undefined' && i18next.t) {
      return i18next.t(key, opts);
    }
    // Fallback English for keys used in this file
    const fallbacks = {
      'mining_getting_proof': 'Mining... getting current proof',
      'mining_difficulty': 'Mining... difficulty={difficulty}, max tries={tries}M',
      'hashing_progress': 'Hashing... {current}k / {total}k | {elapsed}s',
      'block_found': '🎉 BLOCK FOUND!',
      'block_mined': 'Block mined! +{reward} BlockCoin',
      'block_mined_toast': '⛏ Block mined! +{reward} BlockCoin',
      'mining_failed': 'Mining failed',
      'network_error_submit': 'Network error during submit',
      'network_error': 'Network error',
      'mining_error': 'Mining error: {message}',
      'mining_stopped': 'Stopped',
      'amount_empty': 'Amount field is empty',
      'invalid_amount': 'Enter a valid positive number',
      'staked_success': 'Staked! Unlocks at block {block}',
      'unstaked_success': 'Unstaked!',
      'no_active_stakes': 'No active stakes',
      'balance_updated': 'Balance updated',
      'network_stats_updated': 'Network stats updated',
      'no_transactions': 'No transactions yet',
      'failed_to_load': 'Failed to load',
      'lottery_reward': '🎁 Lottery Reward',
      'fee': '💸 Fee',
      'sent': '📤 Sent',
      'received': '📥 Received',
      'block_reward': '⛏ Block Reward',
      'stake_label': '🔒 Stake',
      'unstake_label': '🔓 Unstake',
      'airdrop': '🪂 Airdrop',
      'message_fee': '💬 Message Fee',
      'staking_reward': '💸 Staking Reward',
      'invalid_address': 'Enter a valid 64-character hex address.',
      'positive_amount': 'Enter a positive amount.',
      'sent_success': '✓ Sent!',
      'send_error': 'Error: {error}',
      'hide_qr': '📱 Hide QR Code',
      'show_qr': '📱 Show QR Code',
      'qr_error': 'QR error',
      'address_copied': 'Address copied',
      'nothing_to_copy': 'Nothing to copy',
      'qr_not_ready': 'QR not ready',
      'camera_error': 'Camera error: {error}',
      'address_scanned': '✓ Address scanned',
      'not_valid_wallet_qr': '⚠️ Not a valid wallet QR',
      'fee_label': 'Fee: {fee} BlockCoin',
      'pending_income': 'Pending income: {amount} BlockCoin',
      'apr_label': 'APR: {apr}%',
      'unlock_in': 'unlock in ~{time}',
      'min': 'min',
      'h': 'h',
      'm': 'm'
    };
    let result = fallbacks[key];
    if (result && opts) {
      for (const [k, v] of Object.entries(opts)) {
        result = result.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
      }
    }
    return result || key;
  }

  const MY_ADDRESS = (() => {
    const meta = document.querySelector('meta[name="user-address"]');
    return meta ? meta.content : '';
  })();

  var BLOCKCOIN_SATS = 1_000_000;
  var qrGenerated = false;
  var miningActive = false;
  var POW_MAX_ITERATIONS = 5000000;
  var AUTO_MINING_KEY = 'autoMining'; // ключ для localStorage
  let miningWorker = null;
  // ================== Майнинг ==================
  // ================== Майнинг ==================
  async function startMining() {
    if (miningActive) return;
    miningActive = true;
    // Сохраняем состояние
    localStorage.setItem(AUTO_MINING_KEY, 'true');

    document.getElementById('startMiningBtn').classList.add('hidden');
    document.getElementById('stopMiningBtn').classList.remove('hidden');
    const statusEl = document.getElementById('miningStatus');
    const progressBar = document.getElementById('miningProgress');
    const fillEl = progressBar.querySelector('.mining-progress-bar-fill');
    progressBar.classList.remove('hidden');

    try {
      const proofResp = await fetch('/wallet/last-proof');
      const chainData = await proofResp.json();
      let { last_proof, last_index, difficulty, challenge } = chainData;
      const maxIter = POW_MAX_ITERATIONS;

      const runWorker = () => {
        if (!miningActive) return;
        if (miningWorker) miningWorker.terminate();
        miningWorker = new Worker('/static/js/mining-worker.js');
        const startTime = Date.now();

        // ВАЖНО: Отправляем данные в формате, который ждет новый класс воркера
        miningWorker.postMessage({
          command: 'start',
          payload: { last_proof, last_index, challenge, difficulty, maxIter, startTime }
        });

        miningWorker.onmessage = async (e) => {
          const data = e.data;

          // Обработка прогресса (используем type из нового воркера)
          if (data.type === 'progress') {
            const pct = Math.min(100, Math.floor((data.progress / maxIter) * 100));
            fillEl.style.width = pct + '%';
            const hashrate = data.hashrate || (data.progress / ((Date.now() - startTime) / 1000));
            const etaMin = data.eta && data.eta !== Infinity ? (data.eta / 60).toFixed(1) : '∞';
            statusEl.textContent = `Hashing... ${Math.floor(data.progress/1000)}k / ${Math.floor(maxIter/1000)}k | ${Math.floor(hashrate)} h/s | ETA ${etaMin} min`;
            return;
          }

          // Блок найден
          if (data.type === 'found') {
            const res = await fetch('/wallet/mine', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              // Обязательно добавляем last_index из ответа воркера
              body: JSON.stringify({ proof: data.proof, challenge, last_proof, last_index: data.last_index })
            });
            const result = await res.json();
            if (res.ok) {
              refreshBalance();
              refreshNetworkStats();
              loadTx();
              stopMining();
              setTimeout(() => startMining(), 2000);
            } else if (res.status === 409) {
              const fresh = await fetch('/wallet/last-proof');
              const freshData = await fresh.json();
              last_proof = freshData.last_proof;
              last_index = freshData.last_index;
              challenge = freshData.challenge;
              difficulty = freshData.difficulty;
              runWorker();
            } else {
              stopMining();
            }
          }

          // Решение не найдено (исчерпаны итерации)
          if (data.type === 'not_found') {
            const fresh = await fetch('/wallet/last-proof');
            const freshData = await fresh.json();
            last_proof = freshData.last_proof;
            last_index = freshData.last_index;
            challenge = freshData.challenge;
            difficulty = freshData.difficulty;
            runWorker();
          }
        };
      };
      runWorker();
    } catch (err) {
      console.error(err);
      stopMining();
    }
  }

  function stopMining() {
    miningActive = false;
    localStorage.setItem(AUTO_MINING_KEY, 'false');
    if (miningWorker) {
      // Отправляем команду на мягкую остановку перед уничтожением потока
      miningWorker.postMessage({ command: 'stop' });
      miningWorker.terminate();
      miningWorker = null;
    }
    document.getElementById('startMiningBtn').classList.remove('hidden');
    document.getElementById('stopMiningBtn').classList.add('hidden');
    document.getElementById('miningProgress').classList.add('hidden');
    document.getElementById('miningStatus').textContent = t('mining_stopped');
  }

    // ================== Уведомления о новых блоках через WebSocket ==================
  function initMiningNotifications() {
    if (!window.wsClient) {
      console.warn('WebSocket client not ready, mining notifications disabled');
      const checkInterval = setInterval(() => {
        if (window.wsClient) {
          clearInterval(checkInterval);
          attachListener();
        }
      }, 500);
      setTimeout(() => clearInterval(checkInterval), 10000);
      return;
    }
    attachListener();

    function attachListener() {
      if (!window.wsClient) return;
      const originalOnMessage = window.wsClient.onMessage;
      window.wsClient.onMessage = (data) => {
        if (originalOnMessage) originalOnMessage(data);
        if (data.type === 'new_block' && miningActive) {
          console.log('🔔 New block mined by network, restarting mining...');
          refreshNetworkStats();
          refreshBalance();
          const wasActive = miningActive;
          stopMining();
          if (wasActive) {
            setTimeout(() => startMining(), 500);
          }
        }
      };
      console.log('Mining notifications enabled');
    }
  }

  async function sha256(text) {
    const encoder = new TextEncoder();
    const data = encoder.encode(text);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  // ================== Стейкинг ==================
  async function stake() {
    const amountEl = document.getElementById('stakeAmount');
    const rawAmount = parseFloat(amountEl.value);
    if (!amountEl.value.trim()) {
      window.NotificationManager?.showToast(t('amount_empty'), 'warning');
      return;
    }
    if (isNaN(rawAmount) || rawAmount <= 0) {
      window.NotificationManager?.showToast(t('invalid_amount'), 'warning');
      return;
    }
    const amount = Math.floor(rawAmount * BLOCKCOIN_SATS);
    const res = await fetch('/wallet/stake', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount })
    });
    const data = await res.json();
    const el = document.getElementById('stakeResult');
    el.classList.remove('hidden');
    if (res.ok) {
      el.textContent = t('staked_success', { block: data.unlock_block });
      el.style.color = 'var(--status-success)';
      amountEl.value = '';
      refreshBalance();
      loadStakingInfo();
    } else {
      el.textContent = t('error') + ': ' + (data.error || t('unknown'));
      el.style.color = 'var(--status-error)';
    }
  }

  async function unstake() {
    const res = await fetch('/wallet/unstake', { method: 'POST' });
    const data = await res.json();
    const el = document.getElementById('stakeResult');
    el.classList.remove('hidden');
    if (res.ok) {
      el.textContent = t('unstaked_success');
      el.style.color = 'var(--status-success)';
      refreshBalance();
      loadStakingInfo();
    } else {
      el.textContent = t('error') + ': ' + (data.error || t('unstake_locked'));
      el.style.color = 'var(--status-error)';
    }
  }

  function estimateTimeFromBlock(unlockBlock, currentBlock) {
    const blocksLeft = unlockBlock - currentBlock;
    const avgBlockTimeSeconds = 60;
    const totalMinutes = Math.max(0, Math.floor((blocksLeft * avgBlockTimeSeconds) / 60));
    if (totalMinutes < 60) return `${totalMinutes} ${t('min')}`;
    const hours = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    return `${hours}${t('h')} ${mins}${t('m')}`;
  }

  async function loadStakingInfo() {
    const res = await fetch('/wallet/staking/info');
    const data = await res.json();
    const infoEl = document.getElementById('stakingInfo');
    if (data.stakes && data.stakes.length > 0) {
      let html = '';
      const currentBlock = data.current_block || 0;
      let totalStaked = 0;
      let weightedTime = 0;
      const nowSec = Date.now() / 1000;

      data.stakes.forEach(s => {
        totalStaked += s.amount;
        const elapsed = nowSec - s.start_time;
        weightedTime += s.amount * elapsed;
      });

      const avgElapsed = totalStaked > 0 ? weightedTime / totalStaked : 0;
      const years = avgElapsed / (365 * 24 * 3600);
      const apr = years > 0 ? (data.expected_income / totalStaked) / years * 100 : 0;

      data.stakes.forEach(s => {
        const amount = (s.amount / BLOCKCOIN_SATS).toFixed(6);
        const timeEst = estimateTimeFromBlock(s.unlock_block, currentBlock);
        html += `${amount} BlockCoin (${t('unlock_in', { time: timeEst })})<br>`;
      });

      if (data.expected_income) {
        html += `<span style="color: var(--status-success)">${t('pending_income', { amount: (data.expected_income / BLOCKCOIN_SATS).toFixed(6) })}</span><br>`;
      }
      if (totalStaked > 0 && years > 0) {
        html += `<span style="color: var(--accent)">${t('apr_label', { apr: apr.toFixed(2) })}</span>`;
      }
      infoEl.innerHTML = html;
    } else {
      infoEl.textContent = t('no_active_stakes');
    }
  }

  // ================== Баланс ==================
  async function refreshBalance() {
    const res = await fetch('/wallet/balance');
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById('balanceDisplay').textContent = (data.balance / BLOCKCOIN_SATS).toFixed(6) + ' ';
    window.NotificationManager?.showToast(t('balance_updated'), 'success');
    if (data.staked) {
      document.getElementById('stakedInfo').style.display = 'flex';
      document.getElementById('stakedDisplay').textContent = (data.staked / BLOCKCOIN_SATS).toFixed(6) + ' BlockCoin';
    } else {
      document.getElementById('stakedInfo').style.display = 'none';
    }
  }

  // ================== Глобальная статистика сети ==================
  async function refreshNetworkStats() {
    try {
      const res = await fetch('/wallet/global-stats');
      if (!res.ok) return;
      const data = await res.json();
      const divisor = data.coin_divisor;
      if (data.remaining_supply !== null) {
        document.getElementById('remainingSupply').textContent = (data.remaining_supply / divisor).toFixed(6) + ' ' + data.coin_name;
        document.getElementById('remainingRow').style.display = '';
      } else {
        document.getElementById('remainingSupply').textContent = '∞';
        document.getElementById('remainingRow').style.display = 'none';
      }
      document.getElementById('totalSupply').textContent = (data.total_supply / divisor).toFixed(6) + ' ' + data.coin_name;
      document.getElementById('stakingPoolBalance').textContent = (data.staking_pool_balance / divisor).toFixed(6) + ' ' + data.coin_name;
      document.getElementById('messageFee').textContent = (data.message_fee / divisor).toFixed(6) + ' ' + data.coin_name;
      document.getElementById('blockReward').textContent = (data.block_reward / divisor).toFixed(6) + ' ' + data.coin_name;
      document.getElementById('totalBlocks').textContent = data.total_blocks;
      document.getElementById('totalStaked').textContent = (data.total_staked / divisor).toFixed(6) + ' ' + data.coin_name;
      document.getElementById('difficulty').textContent = data.difficulty;
      document.getElementById('stakingFeeRatio').textContent = (data.staking_fee_ratio * 100).toFixed(1) + '%';
      window.NotificationManager?.showToast(t('network_stats_updated'), 'success');
    } catch (err) {
      console.error('Failed to load network stats:', err);
    }
  }

  // ================== Транзакции ==================
  async function loadTx() {
    const list = document.getElementById('txList');
    list.innerHTML = '<div class="loading">' + t('loading') + '</div>';
    const res = await fetch('/wallet/transactions');
    if (!res.ok) {
      list.innerHTML = '<p class="text-muted text-center">' + t('failed_to_load') + '</p>';
      return;
    }
    const data = await res.json();
    if (!data.transactions.length) {
      list.innerHTML = `<div class="empty-state"><div class="icon">💳</div><p>${t('no_transactions')}</p></div>`;
      return;
    }
    while (list.firstChild) list.removeChild(list.firstChild);

    data.transactions.forEach(tx => {
      const item = document.createElement('div');
      item.className = 'list-item';

      let typeLabel = '';
      let sign = '-';
      if (tx.type === 'reward') {
        typeLabel = t('lottery_reward');
        sign = '+';
      } else if (tx.type === 'fee') {
        typeLabel = t('fee');
        sign = '-';
      } else if (tx.type === 'transfer') {
        if (tx.sender === MY_ADDRESS) {
          typeLabel = t('sent');
          sign = '-';
        } else {
          typeLabel = t('received');
          sign = '+';
        }
      } else if (tx.type === 'block_reward') {
        typeLabel = t('block_reward');
        sign = '+';
      } else if (tx.type === 'stake') {
        typeLabel = t('stake_label');
        sign = '-';
      } else if (tx.type === 'unstake') {
        typeLabel = t('unstake_label');
        sign = '+';
      } else if (tx.type === 'airdrop') {
        typeLabel = t('airdrop');
        sign = '+';
      } else if (tx.type === 'message_fee') {
        typeLabel = t('message_fee');
        sign = '-';
      } else if (tx.type === 'staking_reward') {
        typeLabel = t('staking_reward');
        sign = '+';
      }

      const amount = (tx.amount / BLOCKCOIN_SATS).toFixed(6);
      const timestamp = tx.timestamp ? new Date(tx.timestamp * 1000).toLocaleString() : '';

      const infoDiv = document.createElement('div');
      infoDiv.className = 'info';
      infoDiv.style.flex = '1';
      infoDiv.style.overflow = 'hidden';

      const nameDiv = document.createElement('div');
      nameDiv.className = 'name';
      nameDiv.style.fontWeight = '600';
      nameDiv.style.marginBottom = '4px';
      nameDiv.textContent = typeLabel;

      // === НОВОЕ: Добавляем адрес отправителя или получателя ===
      const addressDiv = document.createElement('div');
      addressDiv.className = 'font-mono text-muted';
      addressDiv.style.fontSize = '11px';
      addressDiv.style.whiteSpace = 'nowrap';
      addressDiv.style.overflow = 'hidden';
      addressDiv.style.textOverflow = 'ellipsis';

      if (tx.type === 'transfer') {
        if (tx.sender === MY_ADDRESS) {
          addressDiv.textContent = '→ ' + tx.recipient; // Если отправили мы, показываем получателя
        } else {
          addressDiv.textContent = '← ' + tx.sender; // Если нам, показываем отправителя
        }
      }
      // =========================================================

      const timeDiv = document.createElement('div');
      timeDiv.className = 'text-muted';
      timeDiv.style.fontSize = '10px';
      timeDiv.style.marginTop = '2px';
      timeDiv.textContent = timestamp;

      infoDiv.appendChild(nameDiv);
      if (addressDiv.textContent) infoDiv.appendChild(addressDiv); // Добавляем адрес
      infoDiv.appendChild(timeDiv);

      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'actions';
      actionsDiv.style.marginLeft = '15px';
      actionsDiv.style.textAlign = 'right';

      const amountSpan = document.createElement('span');
      amountSpan.style.fontWeight = '600';
      amountSpan.style.color = sign === '+' ? 'var(--status-success)' : 'var(--status-warning)';
      amountSpan.textContent = `${sign}${amount} BlockCoin`;
      actionsDiv.appendChild(amountSpan);

      item.appendChild(infoDiv);
      item.appendChild(actionsDiv);
      list.appendChild(item);
    });
  }

  async function sendCoins() {
    const addr = document.getElementById('sendAddress').value.trim().toLowerCase();
    const rawAmount = Number(document.getElementById('sendAmount').value);
    if (!addr || addr.length !== 64 || !/^[a-f0-9]{64}$/.test(addr)) {
      window.NotificationManager?.showToast(t('invalid_address'), 'warning');
      return;
    }
    if (isNaN(rawAmount) || rawAmount <= 0) {
      window.NotificationManager?.showToast(t('positive_amount'), 'warning');
      return;
    }
    const amount = Math.floor(rawAmount * BLOCKCOIN_SATS);
    const res = await fetch('/wallet/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipient: addr, amount })
    });
    const result = await res.json();
    const el = document.getElementById('sendResult');
    el.classList.remove('hidden');
    if (res.ok) {
      el.textContent = t('sent_success');
      el.style.color = 'var(--status-success)';
      document.getElementById('sendAmount').value = '';
      document.getElementById('sendAddress').value = '';
      refreshBalance();
      loadTx();
    } else {
      el.textContent = t('send_error', { error: result.error });
      el.style.color = 'var(--status-error)';
    }
  }

  // ================== QR-код и сканер ==================
  function toggleReceiveQR() {
    const block = document.getElementById('qrBlock');
    const btn = document.getElementById('toggleQRBtn');
    if (!block || !btn) return;

    const isHidden = block.classList.contains('hidden');
    if (isHidden) {
      block.classList.remove('hidden');
      if (!qrGenerated) generateQR();
      btn.innerHTML = t('hide_qr');
    } else {
      block.classList.add('hidden');
      btn.innerHTML = t('show_qr');
    }
  }

  function generateQR() {
    const address = MY_ADDRESS;
    const qrDiv = document.getElementById('qrcode');
    if (!address || address === 'None' || !qrDiv || typeof QRCode === 'undefined') return;
    qrDiv.innerHTML = '';
    try {
      QRManager.generateQRCode(qrDiv, address);

    } catch (e) {
      qrDiv.innerHTML = '<span class="text-muted">' + t('qr_error') + '</span>';
    }
  }

  function copyAddressToClipboard() {
    const addr = document.getElementById('receiveAddress')?.textContent?.trim();
    copyToClipboard(addr, t('address_copied'), t('nothing_to_copy'));
  }

  function downloadQR() {
    const canvas = document.querySelector('#qrcode canvas');
    if (!canvas) return window.NotificationManager?.showToast(t('qr_not_ready'), 'error');
    const link = document.createElement('a');
    link.download = `wallet-qr-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }

  async function shareQR() {
    const address = document.getElementById('receiveAddress')?.textContent?.trim();
    if (!address) return;
    if (navigator.share) {
      try {
        await navigator.share({ title: t('my_wallet_address'), text: address });
      } catch (e) { if (e.name !== 'AbortError') console.error(e); }
    } else {
      copyAddressToClipboard();
    }
  }

  async function copyToClipboard(text, successMsg, errorMsg) {
    if (!text) return window.NotificationManager?.showToast(errorMsg, 'error');
    try {
      await navigator.clipboard.writeText(text);
      window.NotificationManager?.showToast(successMsg, 'success');
    } catch {
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select(); document.execCommand('copy');
        document.body.removeChild(ta);
        window.NotificationManager?.showToast(successMsg, 'success');
      } catch (e) {
        window.NotificationManager?.showToast(errorMsg, 'error');
      }
    }
  }




  // ================== Инициализация ==================
  document.addEventListener('DOMContentLoaded', function() {
    fetch('/wallet/config')
      .then(r => r.json())
      .then(cfg => {
        POW_MAX_ITERATIONS = cfg.pow_max_iterations || 5000000;
        if (!cfg.enable_mining) document.getElementById('miningCard')?.remove();
        if (!cfg.enable_staking) document.getElementById('stakingCard')?.remove();
        window.refreshFeeDisplay();
        if (localStorage.getItem(AUTO_MINING_KEY) === 'true') {
          startMining();
        }
      })
      .catch(e => console.error('Error fetching wallet config:', e));

    refreshBalance();
    loadTx();
    loadStakingInfo();
    refreshNetworkStats();
    initMiningNotifications();
  });

  // ================== Обновление отображения комиссии ==================
  window.refreshFeeDisplay = async function() {
    try {
      const res = await fetch('/wallet/config');
      if (!res.ok) throw new Error('Config not loaded');
      const cfg = await res.json();
      const feeDisplay = document.getElementById('sendFeeDisplay');
      if (feeDisplay) {
        const fee = (cfg.transfer_fee / BLOCKCOIN_SATS).toFixed(6);
        feeDisplay.textContent = t('fee_label', { fee });
      }
    } catch(e) {
      console.warn('Failed to refresh fee display', e);
      const feeDisplay = document.getElementById('sendFeeDisplay');
      if (feeDisplay) feeDisplay.textContent = t('fee_label', { fee: '0.010000' });
    }
  };

  window.startMining = startMining;
  window.stopMining = stopMining;
  window.stake = stake;
  window.unstake = unstake;
  window.sendCoins = sendCoins;
  window.toggleReceiveQR = toggleReceiveQR;
  window.copyAddressToClipboard = copyAddressToClipboard;
  window.downloadQR = downloadQR;
  window.shareQR = shareQR;
  window.openWalletQRScanner = openWalletQRScanner;
  window.forceCloseWalletQRScanner = forceCloseWalletQRScanner;
  window.refreshBalance = refreshBalance;
  window.refreshNetworkStats = refreshNetworkStats;
})();