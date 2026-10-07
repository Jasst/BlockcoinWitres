// context-menu.js — единый движок контекстных меню:
// • правый клик (ПК) • долгое нажатие (тач/стикус) • вибрация • подсветка цели
// Используется для сообщений чата, списка бесед, контактов и групп.
(function () {
    'use strict';
    if (window._contextMenuLoaded) return;
    window._contextMenuLoaded = true;

    const HOLD_MS = 480;          // длительность долгого нажатия
    const MOVE_TOLERANCE = 12;    // пикселей до — считаем свайпом, а не нажатием

    let activeMenu = null;        // { el, cleanup }
    let holdTimer = null;
    let startX = 0, startY = 0;
    let pressedTarget = null;
    let suppressClickUntil = 0;   // блокируем «фантомный» клик после долгого нажатия

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
                    btn.innerHTML = `<span class="ctx-icon">${it.icon || ''}</span><span>${it.label}</span>`;
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

            el.addEventListener('pointerdown', (e) => {
                if (e.pointerType === 'mouse' && e.button !== 0) return; // правый клик обрабатывается выше
                if (e.target.closest('button, a, input, textarea, select, audio, .no-hold')) return;
                pressedTarget = el;
                startX = e.clientX; startY = e.clientY;
                clearTimeout(holdTimer);
                holdTimer = setTimeout(() => {
                    if (!pressedTarget) return;
                    const items = buildItems(el, { x: startX, y: startY });
                    if (!items) return;
                    haptic(15);
                    suppressClickUntil = Date.now() + 700;
                    // подсветка-«отклик»: цель слегка сжимается перед появлением меню
                    el.classList.add('ctx-press');
                    setTimeout(() => el.classList.remove('ctx-press'), 200);
                    this.open(items, { x: startX, y: startY }, { target: el });
                }, HOLD_MS);
            });

            const cancelHold = (e) => {
                if (holdTimer && e && pressedTarget === el) {
                    const dx = Math.abs(e.clientX - startX), dy = Math.abs(e.clientY - startY);
                    if (dx > MOVE_TOLERANCE || dy > MOVE_TOLERANCE) { clearTimeout(holdTimer); holdTimer = null; }
                }
            };
            el.addEventListener('pointermove', cancelHold);
            el.addEventListener('pointerup', () => { clearTimeout(holdTimer); holdTimer = null; });
            el.addEventListener('pointercancel', () => { clearTimeout(holdTimer); holdTimer = null; });
            // если тач-скролл начался — отменяем таймер
            el.addEventListener('scroll', () => { clearTimeout(holdTimer); holdTimer = null; }, { capture: true, passive: true });
        }
    };

    // Глобально: клик вне любого открытого меню закрывает его (страховка)
    document.addEventListener('DOMContentLoaded', () => {
        document.addEventListener('click', (e) => {
            if (activeMenu && !e.target.closest('.ctx-menu')) closeMenu();
        });
    });
})();
