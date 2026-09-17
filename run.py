"""
run.py — Production runner для FastAPI (self-hosted).

ВАЖНО: принудительно 1 воркер. Причина:
  - routes/mcp_auth._sessions — in-memory
  - ai_assistant._assistants — in-memory (CognitiveController)
  - memory_service._services — in-memory
  - MCPToolManager._tool_cache/_rate_limits — in-memory
  - каждый CognitiveController держит свой SSE к MCP под X-User-Id

Multi-worker ломает сессии (401 через раз) и создаёт дубли MCP-подключений
под одним user_id. Масштабирование — отдельная задача (Redis + sticky),
а не подъём workers.

Наружу пускает только nginx — host всегда 127.0.0.1.
"""
import atexit
import logging
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

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(remove_pid)
    save_pid()

    mode = os.getenv('UVICORN_MODE', 'auto')

    # === ИЗМЕНЕНИЕ 1: читаем SERVER_HOST/SERVER_PORT, не HOST/PORT ===
    # В .env уже есть SERVER_HOST=127.0.0.1 — используем его.
    # Если SERVER_HOST не задан — всё равно 127.0.0.1 (наружу пускает nginx).
    host = os.getenv('SERVER_HOST') or os.getenv('HOST') or '127.0.0.1'
    port = int(os.getenv('SERVER_PORT') or os.getenv('PORT', 8000))

    is_prod = os.getenv('FLASK_ENV') == 'production'

    # === ИЗМЕНЕНИЕ 2: workers всегда 1 ===
    # Не читаем SERVER_WORKERS — она тут не применима.
    workers = 1
    reload = (mode == 'dev')

    # === ИЗМЕНЕНИЕ 3: uvloop только если реально установлен ===
    try:
        import uvloop  # noqa: F401
        loop_impl = 'uvloop'
    except ImportError:
        loop_impl = 'asyncio'

    # === ИЗМЕНЕНИЕ 4: страховка от случайного 0.0.0.0 ===
    if host == '0.0.0.0' and not os.getenv('ALLOW_PUBLIC_BIND'):
        print("!! WARNING: SERVER_HOST=0.0.0.0 отключён, ставлю 127.0.0.1.")
        print("!! Если это осознанно — запусти с ALLOW_PUBLIC_BIND=1")
        host = '127.0.0.1'

    print("=" * 60)
    print("🚀  BiChat Messenger Server (FastAPI + Uvicorn)")
    print("=" * 60)
    print(f"   Mode:    {mode}")
    print(f"   Host:    {host}:{port}   (behind nginx)")
    print(f"   Workers: {workers}  (forced 1: in-memory state)")
    print(f"   Loop:    {loop_impl}")
    print(f"   Reload:  {reload}")
    print(f"   Docs:    http://{host}:{port}/api/docs")
    print(f"   PID:     {os.getpid()}")
    print("=" * 60)

    uvicorn.run(
        'main:app',
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        loop=loop_impl,
        http='httptools',
        access_log=not is_prod,
        server_header=False,
        date_header=False,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30,
    )