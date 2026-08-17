self.addEventListener('install', event => {
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(clients.claim());
});

// ✅ SW знает, в каком чате сейчас пользователь
let activeChatId = null;

self.addEventListener('message', event => {
    if (event.data?.type === 'SET_ACTIVE_CHAT') {
        activeChatId = event.data.chatId || null;
    }
    if (event.data?.type === 'CLEAR_ACTIVE_CHAT') {
        activeChatId = null;
    }
});

// ═══════════════════════════════════════════════════════════
// PUSH
// ═══════════════════════════════════════════════════════════
self.addEventListener('push', event => {
    let data = {
        type: 'message',
        title: 'BiChat',
        body: 'New message',
        url: '/chat'
    };

    if (event.data) {
        try {
            const parsed = event.data.json();
            data = { ...data, ...parsed };
        } catch (e) {
            data.body = event.data.text() || '';
        }
    }

    const isCall = data.type === 'incoming_call';
    const tag = isCall ? `call-${data.call_id}` : `msg-${data.chat_id || data.from || Date.now()}`;

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
            const hasVisibleClient = clientList.some(client =>
                client.url.startsWith(self.location.origin) &&
                client.visibilityState === 'visible'
            );

            // Звонки: ВСЕГДА показываем
            if (isCall) {
                return self.registration.showNotification(
                    `📞 ${data.from_name || 'Incoming call'}`,
                    {
                        body: data.body,
                        icon: '/static/icon-192.png',
                        badge: '/static/badge-72.png',
                        data: {
                            url: data.url,
                            type: data.type,
                            call_id: data.call_id || null,
                            from: data.from || null,
                            from_name: data.from_name || null
                        },
                        tag: tag,
                        renotify: true,
                        vibrate: [200, 100, 200, 100, 200],
                        silent: false,
                        requireInteraction: true
                    }
                );
            }

            // Сообщения: проверяем активный чат
            if (hasVisibleClient) {
                // ✅ ИСПРАВЛЕНИЕ: сначала chat_id (работает для групп!), потом from
                const msgChatId = data.chat_id || data.from || null;

                // Пользователь в ЭТОМ чате → не показываем (видно на экране)
                if (activeChatId && msgChatId && activeChatId === msgChatId) {
                    return;
                }

                // Пользователь в другом чате или на экране списка → показываем!
            }

            return self.registration.showNotification(
                data.title,
                {
                    body: data.body,
                    icon: '/static/icon-192.png',
                    badge: '/static/badge-72.png',
                    data: {
                        url: data.url,
                        type: data.type,
                        call_id: data.call_id || null,
                        from: data.from || null,
                        from_name: data.from_name || null,
                        chat_id: data.chat_id || null
                    },
                    tag: tag,
                    renotify: true,
                    silent: false
                }
            );
        })
    );
});

// ═══════════════════════════════════════════════════════════
// NOTIFICATION CLICK
// ═══════════════════════════════════════════════════════════
self.addEventListener('notificationclick', event => {
    event.notification.close();

    const data = event.notification.data || {};
    const type = data.type;
    const callId = data.call_id;

    let targetUrl = data.url || '/chat';
    if (callId && !targetUrl.includes('call_id')) {
        const separator = targetUrl.includes('?') ? '&' : '?';
        targetUrl += `${separator}call_id=${callId}`;
    }

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
            for (const client of clientList) {
                if (client.url.startsWith(self.location.origin)) {
                    client.focus();
                    if (type === 'incoming_call' && callId) {
                        client.postMessage({
                            type: 'open_call',
                            url: targetUrl,
                            call_id: callId,
                            from: data.from,
                            from_name: data.from_name
                        });
                    } else {
                        client.postMessage({
                            type: 'navigate',
                            url: targetUrl
                        });
                    }
                    return;
                }
            }
            return clients.openWindow(targetUrl);
        })
    );
});

// ═══════════════════════════════════════════════════════════
// PUSH SUBSCRIPTION CHANGE
// ═══════════════════════════════════════════════════════════
self.addEventListener('pushsubscriptionchange', event => {
    console.log('Push subscription expired/changed');
    event.waitUntil(
        self.registration.pushManager.getSubscription()
            .then(oldSubscription => {
                const oldEndpoint = oldSubscription ? oldSubscription.endpoint : null;
                const vapidKey = 'BPa5fghsHcpAbmlQTdXg6WzoMC_iPaDMzFY4mc2BUipmno6sLxN6KoSfaZfgUFkh9c0B34XhBvC93WXn92xKlkw';
                return self.registration.pushManager.subscribe({
                    userVisibleOnly: true,
                    applicationServerKey: urlBase64ToUint8Array(vapidKey)
                }).then(newSubscription => {
                    if (!newSubscription) return;
                    return fetch('/push/renew', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            old_endpoint: oldEndpoint,
                            subscription: newSubscription
                        })
                    });
                });
            })
            .catch(err => console.error('Resubscribe failed:', err))
    );
});

function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replace(/\-/g, '+').replace(/_/g, '/');
    const rawData = atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}