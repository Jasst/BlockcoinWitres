/**
 * spa-nav.js — переходы между вкладками без перезагрузки страницы
 *
 * Решает две проблемы:
 * 1) <base target="_blank"> открывал каждую вкладку в НОВОЙ вкладке браузера.
 *    Новая вкладка = пустой sessionStorage = core.js показывал модалку
 *    «Unlock wallet» при КАЖДОМ переходе.
 * 2) Обычные <a href="/chat"> и т.п. вызывали полную перезагрузку:
 *    разрыв WebSocket, повторная загрузка всех скриптов, мигание UI.
 *
 * Как работает: перехватываем клики по внутренним ссылкам, fetch-им страницу,
 * меняем только .content-wrapper, выполняем её скрипты и pushState в историю.
 * WebSocket, мнемоника (sessionStorage) и все глобальные менеджеры живут дальше.
 *
 * ВАЖНО: ai-manager.js намеренно НЕ перезапускается (защита от дублирующихся
 * глобальных обработчиков). Но ui.js при каждом SPA-заходе на /chat сбрасывает
 * window.selectConversation / window.loadConversations, поэтому после
 * выполнения скриптов страницы обёртки ai-manager восстанавливаются вручную —
 * см. restoreGlobalOverrides().
 */
(function () {
  if (window._spaNavLoaded) return;
  window._spaNavLoaded = true;

  // Куда НЕ ходим через SPA — полная загрузка (публичные/служебные страницы)
  const FULL_LOAD_PATHS = ['/', '/index', '/login', '/logout', '/create_wallet'];

  // Скрипты base.html и их CDN — не перезапускаем: их состояние (State, wsClient,
  // NotificationManager, CallManager, ai-manager, DOMPurify...) должно переживать
  // переходы. ВАЖНО: marked и highlight.js сюда НЕ входят — они подключены только
  // в chat.html и ОБЯЗАНЫ выполняться при каждом SPA-заходе на /chat, иначе
  // window.marked/window.hljs не определены и markdown в AI-чате не рендерится
  // (код без подсветки, нет блоков рассуждений). Повторное выполнение безопасно.
  const PROTECTED_SRC = [
    'WebSocketClient.js', 'common.js', 'wordlist.js', 'crypto-client.js',
    'Audio_client.js', 'notification-manager.js', 'mnemonic-manager.js',
    'storage-encryption.js', 'call-manager.js', 'core.js', 'ai-manager.js',
    'i18n.js', 'qr-manager.js',
    'i18next', 'jsqr', 'qrcode', 'purify'
  ];
  const isProtected = (src) => PROTECTED_SRC.some(p => src.includes(p));

  // Гварды страничных скриптов: сбрасываем, чтобы при повторном заходе на
  // страницу её скрипт заново привязал обработчики к свежему DOM
  const PAGE_GUARDS = [
    '_uiLoaded', '_actionsLoaded', '_walletScriptLoaded',
    '_contactsScriptLoaded', '_groupsScriptLoaded', '_profileScriptLoaded',
    '_loginScriptLoaded'
  ];

  let navigating = false;

  // ---------------------------------------------------------------------------
  // Восстановление глобальных обёрток ai-manager.js.
  // ui.js при каждом повторном выполнении делает «window.selectConversation =
  // selectConversation», затирая обёртку ai-manager (без неё selectConversation
  // ('ai_...') открывает AI-сессию как чат с обычным пользователем). То же
  // самое происходит с window.loadConversations (подгрузка AI-сессий в список).
  // Ниже — точные копии этих обёрток из ai-manager.js.
  // ---------------------------------------------------------------------------
  function restoreGlobalOverrides() {
    if (!document.getElementById('aiChatContainer')) return; // не страница чата

    // 1. selectConversation с поддержкой AI-сессий
    if (!window.selectConversation || window.selectConversation._aiWrapper) return void 0;
    const orig = window.selectConversation;
    const wrapped = function (address, name, isGroup) {
      if (address && address.startsWith('ai_')) {
        if (window.State) {
          window.State.currentChatAddress = address;
          window.State.currentChatIsGroup = false;
          window.State.currentChatPartnerAddress = '';
        }
        const mainContainer = document.getElementById('messagesContainer');
        const mainInputArea = document.querySelector('.chat-panel .input-area:not(#aiChatContainer .input-area)');
        const mainChatHeader = document.querySelector('.chat-panel .chat-panel-header:not(#aiChatContainer .chat-panel-header)');
        const aiContainer = document.getElementById('aiChatContainer');
        if (mainContainer) mainContainer.style.display = 'none';
        if (mainInputArea) mainInputArea.style.display = 'none';
        if (mainChatHeader) mainChatHeader.style.display = 'none';
        if (aiContainer) {
          aiContainer.classList.remove('hidden');
          if (typeof window._initAiChatSession === 'function') {
            window._initAiChatSession(address);
          }
        }
        const displayName = name || (address.slice(3, 8) + '…');
        const nameEl = document.getElementById('currentChatName');
        if (nameEl) nameEl.textContent = '🤖 ' + displayName;
        const subtitleEl = document.getElementById('chatSubtitle');
        if (subtitleEl) subtitleEl.textContent = 'AI Assistant';
        if (typeof window._enableChatControls === 'function') window._enableChatControls();
        document.querySelectorAll('.conversation-item').forEach(item => item.classList.remove('active'));
        const activeItem = document.querySelector('.conversation-item[data-address="' + address + '"]');
        if (activeItem) activeItem.classList.add('active');
        if (window.innerWidth < 768 && typeof window.showChatPanel === 'function') {
          window.showChatPanel();
        }
        if (typeof window.adjustMessagesPadding === 'function') {
          setTimeout(window.adjustMessagesPadding, 50);
        }
        return;
      }
      orig(address, name, isGroup);
    };
    wrapped._aiWrapper = true;
    window.selectConversation = wrapped;

    // 2. loadConversations: после отрисовки бесед подгружаем AI-сессии
    if (window.loadConversations && !window.loadConversations._aiWrapper) {
      const origLC = window.loadConversations;
      const wrappedLC = function () {
        const result = origLC.apply(this, arguments);
        setTimeout(() => {
          if (window.loadAiSessionsIntoConversations) window.loadAiSessionsIntoConversations();
        }, 50);
        return result;
      };
      wrappedLC._aiWrapper = true;
      window.loadConversations = wrappedLC;
    }
  }

  // Постраничная инициализация AI-панели чата — копия DOMContentLoaded-логики
  // ai-manager.js (он глобальный и не перезапускается).
  function initAiPanel() {
    if (window.loadAiSessionsIntoConversations) {
      window.loadAiSessionsIntoConversations();
    }
    const headerActions = document.querySelector('.conversations-header .header-actions');
    if (headerActions && !document.getElementById('newAiChatBtn') && window.createAiSession) {
      const btn = document.createElement('button');
      btn.id = 'newAiChatBtn';
      btn.className = 'btn-icon-oval';
      btn.title = 'New AI chat';
      btn.innerHTML = '🤖+';
      btn.onclick = function () {
        const id = window.createAiSession();
        if (window.loadAiSessionsIntoConversations) window.loadAiSessionsIntoConversations();
        const sessions = window.getAiSessions ? window.getAiSessions() : [];
        const found = sessions.find(s => s.id === id);
        window.selectConversation(id, found ? found.name : 'AI Chat', false);
      };
      headerActions.appendChild(btn);
    }
  }

  // Выполняем скрипты распарсенного документа.
  // document-листенеры дедуплицируются по исходнику (защита от двойного
  // навешивания при повторном выполнении страничных скриптов), а обработчики
  // DOMContentLoaded, зарегистрированные во время выполнения, вызываем сразу —
  // реальный DOMContentLoaded уже давно прошёл.
  async function executeScripts(scripts) {
    const onReady = [];
    const seen = (window._spaDocListeners ||= new Set());
    const origAdd = document.addEventListener.bind(document);

    document.addEventListener = function (type, fn, opts) {
      if (type === 'DOMContentLoaded') { onReady.push(fn); return; }
      try {
        const key = type + '|' + String(fn);
        if (seen.has(key)) return;
        seen.add(key);
      } catch (e) {}
      origAdd(type, fn, opts);
    };

    try {
      for (const s of scripts) {
        const src = s.getAttribute('src');
        if (src) {
          if (isProtected(src)) continue; // глобальные уже живут — не трогаем
          try {
            // грузим текст и вставляем как inline: выполнится синхронно,
            // поэтому обработчики DOMContentLoaded попадут в onReady
            const r = await fetch(new URL(src, location.origin), { credentials: 'same-origin' });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const code = await r.text();
            const el = document.createElement('script');
            el.textContent = code;
            document.body.appendChild(el);
            el.remove();
          } catch (e) {
            console.warn('[SPA] не удалось выполнить скрипт:', src, e);
          }
        } else {
          const code = s.textContent || '';
          if (!code.trim()) continue;
          const el = document.createElement('script');
          el.textContent = code;
          document.body.appendChild(el);
          el.remove();
        }
      }
    } finally {
      document.addEventListener = origAdd;
    }

    for (const fn of onReady) {
      try { fn(); } catch (e) { console.error('[SPA] ошибка инициализации страницы:', e); }
    }
  }

  async function navigate(url, push = true) {
    if (navigating) return;
    const target = new URL(url, location.origin);
    if (FULL_LOAD_PATHS.includes(target.pathname)) { location.href = target.href; return; }

    navigating = true;
    try {
      const res = await fetch(target.href, { credentials: 'same-origin' });
      if (!res.ok) { location.href = target.href; return; }
      const html = await res.text();
      const doc = new DOMParser().parseFromString(html, 'text/html');
      const newContent = doc.querySelector('.content-wrapper');
      const curContent = document.querySelector('.content-wrapper');
      if (!newContent || !curContent) { location.href = target.href; return; }

      // 1. Сбрасываем гварды страничных скриптов
      PAGE_GUARDS.forEach(g => { try { delete window[g]; } catch (e) {} });

      // 2. Собираем скрипты ДО замены контента: из контента страницы
      //    + блок {% block scripts %} в конце <body>
      const scripts = [
        ...newContent.querySelectorAll('script'),
        ...Array.from(doc.body.querySelectorAll(':scope > script'))
      ];

      // 3. Заменяем контент (innerHTML не выполняет скрипты — это и нужно)
      curContent.innerHTML = newContent.innerHTML;

      // 4. Выполняем скрипты страницы + её инициализацию
      await executeScripts(scripts);

      // 5. Восстанавливаем глобальные обёртки ai-manager, затираемые ui.js
      restoreGlobalOverrides();

      // 6. Заголовок и активная вкладка навигации
      if (doc.title) document.title = doc.title;
      document.querySelectorAll('.bottom-nav-item').forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === target.pathname);
      });
      document.body.classList.remove('chat-open');

      // 7. Переводим свежий контент
      if (window.localizePage) window.localizePage();

      // 8. Специфика чата: WebSocket уже подключен и onConnect не сработает
      //    повторно — обновляем список бесед, затем (ПОСЛЕ его отрисовки,
      //    иначе innerHTML='' внутри loadConversations сотрёт AI-сессии)
      //    восстанавливаем AI-панель: сессии + кнопку «Новый AI чат».
      if (document.getElementById('conversationsList') && window.loadConversations) {
        Promise.resolve(window.loadConversations()).finally(() => {
          initAiPanel();
          if (window.adjustMessagesPadding) window.adjustMessagesPadding();
        });
      }

      if (push) history.pushState({ spa: 1 }, '', target.pathname + target.search);
      window.scrollTo(0, 0);
    } catch (e) {
      console.error('[SPA] переход не удался, делаю полную загрузку:', e);
      location.href = target.href;
    } finally {
      navigating = false;
    }
  }

  // Перехват кликов по внутренним ссылкам
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest('a[href]');
    if (!a) return;
    if (a.target === '_blank' || a.hasAttribute('download')) return;
    const href = a.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) return;
    let url;
    try { url = new URL(href, location.origin); } catch { return; }
    if (url.origin !== location.origin) return;
    if (FULL_LOAD_PATHS.includes(url.pathname)) return; // пусть браузер грузит полностью
    e.preventDefault();
    if (url.pathname === location.pathname && url.search === location.search) return;
    navigate(url.pathname + url.search);
  });

  window.addEventListener('popstate', () => {
    navigate(location.pathname + location.search, false);
  });

  console.log('✅ SPA navigation enabled');
})();