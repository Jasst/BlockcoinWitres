import multiprocessing
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Сетевые настройки
bind             = "0.0.0.0:8000"  # Слушаем все интерфейсы для Docker
worker_class     = "gevent"
worker_connections = 1000
workers          = multiprocessing.cpu_count() * 2 + 1
timeout          = 120
graceful_timeout = 30
keepalive        = 5

# Перезапуск воркеров для предотвращения утечек памяти
max_requests     = 1000
max_requests_jitter = 100

# Логирование
accesslog        = "-"  # Вывод в stdout (для Docker)
errorlog         = "-"  # Вывод в stderr (для Docker)
loglevel         = "info"
pidfile          = None  # Не создаём pid файл в Docker

# WebSocket поддержка
forwarded_allow_ips = "*"