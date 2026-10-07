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
    let suppressClickUntil = 0;   // блокируем «фантомный» клик после долгого нажатия

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
        }
    };

    // Глобально: клик вне любого открытого меню закрывает его (страховка)
    document.addEventListener('click', (e) => {
        if (activeMenu && !e.target.closest('.ctx-menu')) closeMenu();
    });
    // Safety net: any native context menu opened on a bound element (e.g. text
    // selection bubble on some Android keyboards) closes our custom menu too.
    document.addEventListener('contextmenu', () => { if (activeMenu) closeMenu(); });
})();
