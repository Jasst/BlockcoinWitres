// app-init.js — Инициализация приложения: SW + WebSocket + Push
// Подключать в base.html последним скриптом перед </body>
// <script src="/static/app-init.js"></script>

(function () {
    'use strict';

    // ─────────────────────────────────────────────────────────────────────
    // 1. РЕГИСТРАЦИЯ SERVICE WORKER
    //    Делаем это СРАЗУ при загрузке страницы, без ожидания разрешения
    //    на уведомления. SW нужен для push независимо от разрешений.
    //    БЕЗ ЭТОГО navigator.serviceWorker.ready никогда не резолвится →
    //    push подписка невозможна → уведомления не приходят никогда.
    // ─────────────────────────────────────────────────────────────────────
    async function registerSW() {
        if (!('serviceWorker' in navigator)) return null;

        try {
            // Проверяем есть ли уже регистрация
            const existing = await navigator.serviceWorker.getRegistration();
            if (existing) {
                // Обновляем SW если есть новая версия
                existing.update().catch(() => {});
                return existing;
            }

            const reg = await navigator.serviceWorker.register('/sw.js', {
                scope: '/',
                // updateViaCache: 'none' — всегда проверяем новую версию SW
                updateViaCache: 'none'
            });
            console.log('[SW] Registered ✓ scope:', reg.scope);
            return reg;
        } catch (e) {
            console.error('[SW] Registration failed:', e.message);
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 2. ИНИЦИАЛИЗАЦИЯ WEBSOCKET + HEARTBEAT + POLLING
    //    Запускаем после загрузки DOM
    // ─────────────────────────────────────────────────────────────────────
    async function startApp() {
        // Проверяем что мы на защищённой странице (пользователь залогинен)
        const userAddress = document.querySelector('meta[name="user-address"]')?.content;
        if (!userAddress || userAddress === 'None' || userAddress === '') {
            console.log('[App] No user address — skipping app init (login page)');
            return;
        }

        // Регистрируем SW сразу
        await registerSW();

        // Запускаем WebSocket
        if (window.initWebSocket) {
            await window.initWebSocket().catch(e => console.warn('[WS] Init error:', e));
        }

        // Heartbeat + статусы
        if (window.startHeartbeat) window.startHeartbeat();
        if (window.startUserStatusPolling) window.startUserStatusPolling();

        // Push: инициализируем если разрешение уже дано
        // (если нет — пользователь жмёт кнопку "Enable" в профиле)
        if (Notification.permission === 'granted' && window.initPushNotifications) {
            window.initPushNotifications().catch(e => console.warn('[Push] Init error:', e));
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 3. ЗАПУСК
    // ─────────────────────────────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startApp);
    } else {
        // DOM уже готов (скрипт подключён в конце body)
        startApp();
    }

})();