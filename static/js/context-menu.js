// context-menu.js — единый движок контекстных меню:
// • правый клик (ПК) • долгое нажатие (тач/стикус) • вибрация • подсветка цели
// Используется для сообщений чата, списка бесед, контактов и групп.
(function () {
    'use strict';
    if (window._contextMenuLoaded) return;
    window._contextMenuLoaded = true;

    const HOLD_MS = 450;          // длительность долгого нажатия
    const MOVE_TOLERANCE = 12;    // пикселей до — считаем свайпом, а не нажатием

    let activeMenu = null;        // { el, cleanup }
    // Per-element hold state (a single global timer broke the menu when two
    // elements were touched/pressed close together - mobile bug).
    const holdState = new WeakMap();
    let suppressClickUntil = 0;   // блокируем contextmenu сразу после долгого нажатия
    // Флаг вместо окна по времени: после долгого нажатия браузер может прислать click
    // позже любого окна (при долгом удержании). Гасим его один раз, до следующего касания.
    let holdFired = false;

    function sanitizeIcon(raw) {
        raw = String(raw || '');
        if (!raw.trim()) return '';
        // Only ever render SVG markup as an icon; any emoji/text passed by
        // mistake is dropped so a menu item can never show "two icons".
        const start = raw.indexOf('<svg');
        if (start === -1) return '';
        const end = raw.lastIndexOf('</svg>');
        if (end === -1) return '';
        return raw.slice(start, end + 6);
    }

    // ---------- Вспомогательные ----------
    function haptic(ms) {
        try { if (navigator.vibrate) navigator.vibrate(ms || 12); } catch (e) {}
    }

    function closeMenu() {
        if (!activeMenu) return;
        const { el, target } = activeMenu;
        if (target) target.classList.remove('ctx-target-active');
        if (typeof activeMenu.cleanup === 'function') activeMenu.cleanup();
        el.classList.remove('active');
        // лёгкая анимация закрытия
        setTimeout(() => el.remove(), 140);
        activeMenu = null;
        document.removeEventListener('pointerdown', onDocPointerDown, true);
    }

    function onDocPointerDown(e) {
        if (activeMenu && !e.target.closest('.ctx-menu')) closeMenu();
    }

    function positionMenu(menu, x, y) {
        menu.style.left = x + 'px';
        menu.style.top = y + 'px';
        const rect = menu.getBoundingClientRect();
        const vw = window.innerWidth, vh = window.innerHeight;
        if (rect.right > vw - 8)  menu.style.left = Math.max(8, x - rect.width) + 'px';
        if (rect.bottom > vh - 8) menu.style.top = Math.max(8, y - rect.height) + 'px';
    }

    // ---------- Мобильный жест: свайп в сторону открывает меню ----------
    // Только для устройств с тачскрином (pointer: coarse). На ПК ничего не меняется:
    // там остаётся правый клик. Долгое нажатие на телефоне тоже сохранено.
    function bindSwipe(el, buildItems) {
        const SWIPE_MIN = 56;   // минимальный сдвиг, px, чтобы открыть меню
        const LOCK = 10;        // сначала определяем направление жеста
        let sx = 0, sy = 0, dx = 0, active = false, locked = false, horizontal = false;

        el.addEventListener('touchstart', (e) => {
            if (!e.touches || e.touches.length !== 1) { active = false; return; }
            if (e.target.closest('button, a, input, textarea, select, audio, .no-hold')) { active = false; return; }
            const t0 = e.touches[0];
            sx = t0.clientX; sy = t0.clientY;
            dx = 0; locked = false; horizontal = false; active = true;
        }, { passive: true });

        el.addEventListener('touchmove', (e) => {
            if (!active || !e.touches || !e.touches.length) return;
            const t0 = e.touches[0];
            const x = t0.clientX - sx, y = t0.clientY - sy;
            if (!locked && (Math.abs(x) > LOCK || Math.abs(y) > LOCK)) {
                locked = true;
                horizontal = Math.abs(x) > Math.abs(y) * 1.5;
            }
            if (locked && horizontal) {
                dx = x;
                // лёгкое «подтягивание» карточки за пальцем, максимум 90px
                el.style.transition = 'none';
                el.style.transform = 'translateX(' + Math.max(-90, Math.min(90, dx * 0.5)) + 'px)';
            }
        }, { passive: true });

        function reset() {
            active = false;
            el.style.transition = 'transform 0.2s ease';
            el.style.transform = '';
        }

        el.addEventListener('touchend', () => {
            if (!active) return;
            const doOpen = locked && horizontal && Math.abs(dx) >= SWIPE_MIN;
            reset();
            if (!doOpen) return;
            // Клик, который браузер пришлёт после свайпа, не должен открыть чат
            holdFired = true;
            haptic(12);
            const items = buildItems(el, { x: sx, y: sy });
            if (items) ContextMenu.open(items, { x: sx, y: sy }, { target: el });
        });
        el.addEventListener('touchcancel', reset);
    }

    // ---------- Публичный API ----------
    window.ContextMenu = {
        /**
         * Открыть меню.
         * items: [{ icon, label, danger, hidden, onClick }] | HTMLElement
         * anchor: {x,y} координаты экрана или HTMLElement (меню привяжется к нему)
         */
        open(items, anchor, opts = {}) {
            closeMenu();

            const menu = document.createElement('div');
            menu.className = 'ctx-menu active';
            menu.setAttribute('role', 'menu');

            if (Array.isArray(items)) {
                for (const it of items) {
                    if (!it) continue;
                    if (it.separator) {
                        const sep = document.createElement('div');
                        sep.className = 'ctx-separator';
                        menu.appendChild(sep);
                        continue;
                    }
                    if (it.hidden) continue;
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'ctx-item' + (it.danger ? ' danger' : '');
                    btn.setAttribute('role', 'menuitem');
                    // Одна иконка: берём ТОЛЬКО <svg>-разметку; эмодзи/текст вне тегов
                    // отбрасываются — пункт меню не может показать «две иконки».
                    const iconHtml = sanitizeIcon(it.icon);
                    const labelSpan = document.createElement('span');
                    labelSpan.className = 'ctx-label';
                    labelSpan.textContent = it.label || '';
                    btn.innerHTML = `<span class="ctx-icon">${iconHtml}</span>`;
                    btn.appendChild(labelSpan);
                    // Пункты без текста (например, «Копировать»/«Удалить» в чате)
                    // остаются доступными: подпись уезжает в title/aria-label.
                    const a11yLabel = it.title || it.label || '';
                    if (a11yLabel) {
                        btn.title = a11yLabel;
                        btn.setAttribute('aria-label', a11yLabel);
                    }
                    if (!labelSpan.textContent) btn.classList.add('icon-only');
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        closeMenu();
                        try { it.onClick && it.onClick(e); } catch (err) { console.error('ctx action:', err); }
                    });
                    menu.appendChild(btn);
                }
            } else if (items instanceof HTMLElement) {
                menu.appendChild(items);
            }

            document.body.appendChild(menu);

            let ax, ay;
            if (anchor instanceof HTMLElement) {
                const r = anchor.getBoundingClientRect();
                ax = r.left + Math.min(r.width / 2, 120);
                ay = r.bottom + 6;
            } else {
                ax = anchor.x; ay = anchor.y;
            }
            positionMenu(menu, ax, ay);

            activeMenu = { el: menu, target: opts.target || null };
            if (opts.target) opts.target.classList.add('ctx-target-active');

            // Esc закрывает меню
            const onKey = (e) => { if (e.key === 'Escape') closeMenu(); };
            document.addEventListener('keydown', onKey);
            activeMenu.cleanup = () => document.removeEventListener('keydown', onKey);

            requestAnimationFrame(() => menu.classList.add('ctx-visible'));
            document.addEventListener('pointerdown', onDocPointerDown, true);
            return menu;
        },
        close: closeMenu,
        isOpen: () => !!activeMenu,

        /**
         * Навесить обработчики на элемент: правый клик + долгое нажатие.
         * buildItems(targetEl, {x,y}) -> массив items
         */
        bind(el, buildItems, opts = {}) {
            el.addEventListener('contextmenu', (e) => {
                if (Date.now() < suppressClickUntil) return;
                e.preventDefault();
                e.stopPropagation();
                haptic(8);
                const items = buildItems(el, { x: e.clientX, y: e.clientY });
                if (items) this.open(items, { x: e.clientX, y: e.clientY }, { target: el });
            }, { passive: false });

            // --- Long press: Pointer Events first, touch/mouse fallbacks for
            // older mobile browsers where pointer events are unreliable. ---
            function getState() {
                let s = holdState.get(el);
                if (!s) { s = { timer: null, x: 0, y: 0 }; holdState.set(el, s); }
                return s;
            }
            function cancelHold() {
                const s = holdState.get(el);
                if (s && s.timer) { clearTimeout(s.timer); s.timer = null; }
            }
            function startHold(x, y) {
                const s = getState();
                cancelHold();
                s.x = x; s.y = y;
                s.timer = setTimeout(() => {
                    s.timer = null;
                    const items = buildItems(el, { x: s.x, y: s.y });
                    if (!items) return;
                    haptic(15);
                    suppressClickUntil = Date.now() + 700;
                    holdFired = true;
                    // подсветка-«отклик»: цель слегка сжимается перед появлением меню
                    el.classList.add('ctx-press');
                    setTimeout(() => el.classList.remove('ctx-press'), 200);
                    ContextMenu.open(items, { x: s.x, y: s.y }, { target: el });
                }, HOLD_MS);
            }
            function movedTooFar(x, y) {
                const s = holdState.get(el);
                if (!s || !s.timer) return false;
                if (Math.abs(x - s.x) > MOVE_TOLERANCE || Math.abs(y - s.y) > MOVE_TOLERANCE) {
                    cancelHold();
                    return true;
                }
                return false;
            }

            el.addEventListener('pointerdown', (e) => {
                if (e.pointerType === 'mouse' && e.button !== 0) return; // правый клик обрабатывается выше
                if (e.target.closest('button, a, input, textarea, select, audio, .no-hold')) return;
                startHold(e.clientX, e.clientY);
            });
            el.addEventListener('pointermove', (e) => { movedTooFar(e.clientX, e.clientY); });
            el.addEventListener('pointerup', cancelHold);
            el.addEventListener('pointercancel', cancelHold);

            // Fallbacks (iOS Safari <13 / old Android WebView): touch events
            el.addEventListener('touchstart', (e) => {
                if (e.target.closest('button, a, input, textarea, select, audio, .no-hold')) return;
                const t0 = e.touches[0];
                if (!t0) return;
                startHold(t0.clientX, t0.clientY);
            }, { passive: true });
            el.addEventListener('touchmove', (e) => {
                const t0 = e.touches[0];
                if (t0) movedTooFar(t0.clientX, t0.clientY);
            }, { passive: true });
            el.addEventListener('touchend', cancelHold);
            el.addEventListener('touchcancel', cancelHold);

            // если тач-скролл начался — отменяем таймер
            el.addEventListener('scroll', cancelHold, { capture: true, passive: true });

            // Android long-press produces a NATIVE contextmenu event ~600ms
            // after touchstart; our custom menu already opened at HOLD_MS.
            // Without this suppression both menus would stack on screen.
            el.addEventListener('contextmenu', (e) => {
                if (Date.now() < suppressClickUntil || activeMenu) e.preventDefault();
            });

            // Свайп — только на сенсорных устройствах
            const coarse = !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
            if (coarse) bindSwipe(el, buildItems);
        }
    };

    // Глобально: клик вне открытого меню закрывает его (страховка).
    // Фаза capture: браузер шлёт «хвостовой» click после долгого нажатия —
    // глушим его ДО onclick беседы/сообщения, иначе меню закрывается сразу.
    // Касание вне открытого меню только закрывает его: клик под ним гасим.
    // Новое касание сбрасывает флаг, поэтому «лишний» клик не съест следующий тап.
    function armClickGuard(e) {
        holdFired = !!(activeMenu && !e.target.closest('.ctx-menu'));
    }
    document.addEventListener('pointerdown', armClickGuard, true);
    document.addEventListener('touchstart', armClickGuard, { capture: true, passive: true });
    document.addEventListener('click', (e) => {
        const inMenu = !!e.target.closest('.ctx-menu');
        if (!inMenu && holdFired) {
            holdFired = false;
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        if (activeMenu && !inMenu) closeMenu();
    }, true);
    // Safety net: any native context menu opened on a bound element (e.g. text
    // selection bubble on some Android keyboards) closes our custom menu too.
    document.addEventListener('contextmenu', () => { if (activeMenu) closeMenu(); });
})();