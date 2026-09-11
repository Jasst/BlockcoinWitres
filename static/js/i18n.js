// static/js/i18n.js
(function() {
    if (typeof i18next === 'undefined') {
        console.error('i18next not loaded');
        document.body.classList.add('i18n-ready');
        return;
    }
    if (typeof i18nextHttpBackend === 'undefined') {
        console.error('i18nextHttpBackend not loaded');
        document.body.classList.add('i18n-ready');
        return;
    }
    if (typeof i18nextBrowserLanguageDetector === 'undefined') {
        console.error('i18nextBrowserLanguageDetector not loaded');
        document.body.classList.add('i18n-ready');
        return;
    }

    i18next
        .use(i18nextHttpBackend)
        .use(i18nextBrowserLanguageDetector)
        .init({
            backend: { loadPath: '/static/locales/{{lng}}/translation.json' },
            fallbackLng: 'en',
            supportedLngs: ['en', 'ru'],
            detection: { order: ['localStorage', 'navigator'], lookupLocalStorage: 'app_lang' },
            interpolation: { escapeValue: false }
        }, (err) => {
            if (err) console.error('i18next init error', err);
            const storedLang = localStorage.getItem('app_lang') || 'en';
            i18next.loadLanguages(storedLang, () => {
                i18next.changeLanguage(storedLang, () => {
                    localizePage();
                    document.body.classList.add('i18n-ready');
                    document.dispatchEvent(new CustomEvent('i18next:initialized'));
                });
            });
        });

    function localizePage() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            if (el.id === 'clearCountdown') return;
            const key = el.getAttribute('data-i18n');
            const isInputLike = el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';
            if (isInputLike) {
                el.placeholder = i18next.t(key);
            } else {
                el.innerHTML = i18next.t(key);
            }
        });
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = i18next.t(key);
        });
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = i18next.t(key);
        });
        document.title = i18next.t('app_name');
        if (window.refreshUIText) window.refreshUIText();

        const langToggle = document.getElementById('langToggle');
        if (langToggle) {
            langToggle.textContent = i18next.language === 'ru' ? 'RU' : 'EN';
        }

        localStorage.setItem('app_lang', i18next.language);

        // Обновляем тексты в CallManager (синхронно после обновления страницы)
        if (window.CallManager && typeof window.CallManager.updateLocalizedTexts === 'function') {
            window.CallManager.updateLocalizedTexts();
        }
    }

    window.changeLanguage = (lng) => {
        if (lng !== 'en' && lng !== 'ru') return;
        if (i18next.language === lng) return;
        localStorage.setItem('app_lang', lng);
        i18next.loadLanguages(lng, () => {
            i18next.changeLanguage(lng, () => {
                localizePage();
                if (window.loadConversations) window.loadConversations();
                if (window.loadContacts) window.loadContacts();
                if (window.loadGroups) window.loadGroups();
                if (window.refreshBalance) window.refreshBalance();
                if (window.refreshNetworkStats) window.refreshNetworkStats();
                if (window.refreshFeeDisplay) window.refreshFeeDisplay();

                // 🔧 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: небольшая задержка для обновления модалки звонка
                setTimeout(() => {
                    if (window.CallManager && typeof window.CallManager.updateLocalizedTexts === 'function') {
                        window.CallManager.updateLocalizedTexts();
                    }
                }, 50);
            });
        });
    };

    window.localizePage = localizePage;
})();