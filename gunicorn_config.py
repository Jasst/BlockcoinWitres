import multiprocessing
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

bind             = "127.0.0.1:8000"
worker_class     = "gevent"
worker_connections = 1000
workers          = multiprocessing.cpu_count() * 2 + 1
timeout          = 120
max_requests     = 1000
max_requests_jitter = 100
accesslog        = os.path.join(BASE_DIR, "access.log")
errorlog         = os.path.join(BASE_DIR, "error.log")
loglevel         = "warning"
pidfile          = os.path.join(BASE_DIR, "app.pid")