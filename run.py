"""
run.py — Production runner using Uvicorn (replaces Waitress)
Supports graceful shutdown and PID file for process managers.

Usage:
    python run.py                     # auto mode
    UVICORN_MODE=dev python run.py    # 1 worker, reload
    UVICORN_MODE=stable python run.py # conservative workers
    UVICORN_MODE=max python run.py    # maximum workers
"""
import atexit
import logging
import multiprocessing
import os
import signal
import sys

logger = logging.getLogger(__name__)

PID_FILE = os.path.join(os.path.dirname(__file__), 'app.pid')


def save_pid():
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        print(f"PID saved: {os.getpid()}")
    except Exception as e:
        print(f"Could not save PID: {e}")


def remove_pid():
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


def graceful_shutdown(signum, frame):
    print("\nReceived shutdown signal, stopping gracefully...")
    remove_pid()
    sys.exit(0)


if __name__ == '__main__':
    import uvicorn

    signal.signal(signal.SIGINT,  graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(remove_pid)
    save_pid()

    cpu_count = multiprocessing.cpu_count()
    mode      = os.getenv('UVICORN_MODE', 'auto')
    port      = int(os.getenv('PORT', 8000))
    host      = os.getenv('HOST', '127.0.0.1')
    is_prod   = os.getenv('FLASK_ENV') == 'production'

    # Worker / thread counts
    if mode == 'dev':
        workers = 1
        reload  = True
    elif mode == 'stable':
        workers = max(2, cpu_count // 2)
        reload  = False
    elif mode == 'max':
        workers = min(cpu_count * 2, 16)
        reload  = False
    else:  # auto
        workers = min(max(cpu_count, 2), 8)
        reload  = not is_prod

    print("=" * 60)
    print("🚀  BiChat Messenger Server (FastAPI + Uvicorn)")
    print("=" * 60)
    print(f"   Mode:    {mode}")
    print(f"   Host:    {host}:{port}")
    print(f"   Workers: {workers}")
    print(f"   Reload:  {reload}")
    print(f"   Docs:    http://{host}:{port}/api/docs")
    print(f"   PID:     {os.getpid()}")
    print("=" * 60)

    if workers > 1 and not reload:
        # Multi-process mode (production)
        uvicorn.run(
            'main:app',
            host=host,
            port=port,
            workers=workers,
            loop='uvloop',
            http='httptools',
            access_log=not is_prod,
            server_header=False,
            date_header=False,
        )
    else:
        # Single-process mode (dev / reload)
        uvicorn.run(
            'main:app',
            host=host,
            port=port,
            reload=reload,
            loop='asyncio',
            access_log=True,
            server_header=False,
        )