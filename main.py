"""
main.py — FastAPI-приложение (PostgreSQL + WebSocket)
"""
import asyncio
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from GCN.config_ai import GENERATED_IMAGES_DIR
from config import CONFIG, SECRET_KEY, STATIC_FOLDER, UPLOAD_FOLDER
from database import init_db, close_db, Blockchain
from setup import setup_logging, get_rate_limit_stats
from services.wallet import init_wallet_service
from mcp_server_blockcoin import mcp
from routes.ai_assistant import start_global_merge_task, init_global_mcp

setup_logging()
logger = logging.getLogger(__name__)

# Явно устанавливаем уровень логирования для модуля с MCP
logging.getLogger("routes.ai_assistant").setLevel(logging.INFO)
logging.getLogger("GCN.mcp_client_manager").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BiChat server (PostgreSQL + WebSocket)...")
    await init_db()
    blockchain = Blockchain()
    app.state.blockchain = blockchain
    init_wallet_service(blockchain)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info("BiChat server started ✅")

    # =====================================================================
    # Запуск фонового cleanup WebSocket-менеджера.
    # РАНЬШЕ это делалось в ConnectionManager.__init__ через
    # asyncio.create_task(...) — но конструктор вызывается на уровне модуля
    # (`manager = ConnectionManager()` в конце routes/ws.py), то есть ДО
    # того, как uvicorn создаст event loop. Это приводило к падению
    # "RuntimeError: no running event loop" при импорте main:app.
    # Теперь запуск происходит здесь — loop гарантированно живой.
    # =====================================================================
    from routes.ws import manager as ws_manager
    ws_manager.start_cleanup_task()
    logger.info("🧹 WebSocket cleanup task started")

    # Запускаем фоновое коллективное обучение AI
    from routes.ai_assistant import start_global_merge_task
    start_global_merge_task()
    logger.info("🌍 Global AI learning task started")

    # =====================================================================
    # MCP: серверная часть работает, клиент должен подключиться к ней.
    #
    # ВАЖНО: init_global_mcp() ходит HTTP-запросом на URL из
    # mcp_servers.json (по умолчанию https://blockcoin.ru/mcp/). Этот URL
    # обслуживается ЭТИМ ЖЕ процессом, поэтому вызывать его синхронно
    # внутри lifespan нельзя: uvicorn не начнёт принимать HTTP до yield,
    # клиент будет ждать ответа до таймаута, и anyio-скоупы внутри
    # mcp.session_manager.run() развалятся с
    # "RuntimeError: Attempted to exit a cancel scope that isn't the
    #  current task's current cancel scope".
    #
    # Решение: запускаем init_global_mcp() как ФОНОВЫЙ таск. Lifespan
    # быстро доходит до yield, uvicorn начинает слушать, и уже после
    # этого клиент MCP успешно подключается к своему же серверу.
    #
    # Для локальной разработки можно переопределить конфиг через env:
    #   PowerShell:  $env:MCP_SERVERS_CONFIG = "E:\BlockcoinWitres\mcp_servers.local.json"
    #   где url = "http://127.0.0.1:8000/mcp/" — не стучаться через
    #   публичный домен/прокси.
    # =====================================================================
    async with mcp.session_manager.run():

        async def _deferred_init_global_mcp():
            # Небольшая пауза, чтобы uvicorn успел открыть порт после yield.
            # 1.5с хватает с запасом на типичном железе; при медленном старте
            # (первая загрузка модели, холодный кэш) увеличь до 3–5с.
            await asyncio.sleep(1.5)
            try:
                await init_global_mcp()
                logger.info("🌐 Global MCP manager initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize global MCP: {e}", exc_info=True)

        init_task = asyncio.create_task(
            _deferred_init_global_mcp(), name="mcp-init-global"
        )

        try:
            yield
        finally:
            # Аккуратная остановка фоновой инициализации при shutdown
            if not init_task.done():
                init_task.cancel()
                try:
                    await init_task
                except asyncio.CancelledError:
                    pass

            # Остановка WebSocket cleanup
            try:
                from routes.ws import manager as ws_manager
                await ws_manager.stop_cleanup_task()
            except Exception as e:
                logger.warning(f"Ошибка при остановке WebSocket cleanup: {e}")

    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title='BiChat Messenger API',
    version='3.0.0-pg-ws',
    lifespan=lifespan,
    docs_url='/api/docs',
    redoc_url='/api/redoc',
    openapi_url='/api/openapi.json',
)

# CORS middleware - настройте allowed_origins для продакшена
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost,http://127.0.0.1').split(','),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie='__Secure-session',
    max_age=CONFIG['SESSION_LIFETIME'],
    https_only=os.getenv('FLASK_ENV') == 'production',
    same_site='lax',
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={'error': 'Internal server error'})


if os.path.isdir(STATIC_FOLDER):
    app.mount('/static', StaticFiles(directory=STATIC_FOLDER), name='static')
if os.path.isdir(UPLOAD_FOLDER):
    app.mount('/uploads', StaticFiles(directory=UPLOAD_FOLDER), name='uploads')

app.mount('/generated_images', StaticFiles(directory=str(GENERATED_IMAGES_DIR)), name='generated_images')
app.mount("/mcp", mcp.streamable_http_app())


@app.get('/sw.js', include_in_schema=False)
async def serve_sw():
    sw_path = os.path.join(STATIC_FOLDER, 'sw.js')
    return FileResponse(
        sw_path,
        media_type='application/javascript',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Service-Worker-Allowed': '/',
        }
    )


@app.get('/manifest.json', include_in_schema=False)
async def serve_manifest():
    manifest_path = os.path.join(STATIC_FOLDER, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise HTTPException(404, 'manifest.json not found')
    return FileResponse(manifest_path, media_type='application/manifest+json',
                        headers={'Cache-Control': 'public, max-age=86400'})


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(STATIC_FOLDER, 'favicon.ico')
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type='image/x-icon')
    raise HTTPException(204)


from routes.auth import router as auth_router
from routes.messages import router as messages_router
from routes.contacts import router as contacts_router
from routes.groups import router as groups_router
from routes.wallet import router as wallet_router
from routes.files import router as files_router
from routes.status import router as status_router
from routes.ai_assistant import router as ai_router
from routes.ws import router as ws_router
from routes.push import router as push_router
from routes.calls import router as calls_router

app.include_router(calls_router)
app.include_router(auth_router)
app.include_router(messages_router)
app.include_router(contacts_router)
app.include_router(groups_router)
app.include_router(wallet_router)
app.include_router(files_router)
app.include_router(status_router)
app.include_router(ai_router)
app.include_router(ws_router)
app.include_router(push_router)


@app.middleware('http')
async def add_cache_headers(request: Request, call_next):
    if request.url.path.startswith("/mcp"):
        try:
            return await call_next(request)
        except RuntimeError as e:
            if "No response returned" in str(e):
                return JSONResponse(status_code=204, content={})
            raise
        except Exception as e:
            logger.error(f"MCP error: {e}")
            return JSONResponse(status_code=500, content={"error": "MCP internal error"})

    try:
        response = await call_next(request)
    except RuntimeError as e:
        if "No response returned" in str(e):
            return JSONResponse(status_code=204, content={})
        raise

    if request.url.path in ['/', '/login', '/create_wallet']:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.get('/health', tags=['health'])
async def health_check(request: Request):
    blockchain: Blockchain = request.app.state.blockchain
    db_health = await blockchain.health_check()
    from routes.ws import manager
    return {
        'status': 'ok' if db_health.get('status') == 'healthy' else 'degraded',
        'database': db_health,
        'rate_limits': get_rate_limit_stats(),
        'websocket': await manager.get_stats(),
    }


@app.get('/health/db', tags=['health'])
async def health_db(request: Request):
    return await request.app.state.blockchain.health_check()


@app.get('/health/performance', tags=['health'])
async def health_performance(request: Request):
    return await request.app.state.blockchain.get_performance_stats()


@app.get('/health/notifier', tags=['health'])
async def health_notifier():
    from services.notifier import message_notifier
    return {'status': 'ok', 'stats': await message_notifier.get_stats()}