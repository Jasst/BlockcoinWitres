"""
routes/files.py — загрузка зашифрованных файлов (изображения/аудио)
"""
import logging
import os
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from config import UPLOAD_FOLDER, MAX_ENCRYPTED_FILE_SIZE
from dependencies import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(tags=['files'])

@router.post('/upload_encrypted')
async def upload_encrypted(
    file: UploadFile = File(...),
    address: str = Depends(require_auth),
):
    """Принимает уже зашифрованный клиентом файл (AES-GCM), сохраняет на диск."""
    content = await file.read()
    if len(content) > MAX_ENCRYPTED_FILE_SIZE:
        raise HTTPException(413, f'File too large (max {MAX_ENCRYPTED_FILE_SIZE//1024//1024} MB)')

    safe_name = uuid.uuid4().hex
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    with open(filepath, 'wb') as f:
        f.write(content)

    logger.info(f"Encrypted file uploaded by {address[:16]}..., size {len(content)} bytes")
    return {'file_url': f"/uploads/{safe_name}"}

@router.get('/uploads/{filename}')
async def serve_upload(filename: str, address: str = Depends(require_auth)):
    """Отдаёт зашифрованный файл только авторизованным пользователям."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, 'File not found')
    return FileResponse(filepath, media_type='application/octet-stream')