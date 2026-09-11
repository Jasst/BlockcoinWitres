"""
models.py — Pydantic-модели запросов/ответов (замена marshmallow)
"""
from typing import List, Optional
from pydantic import BaseModel, field_validator
from config import MAX_MESSAGE_PAYLOAD_SIZE
import json

# =============================================================================
# Auth
# =============================================================================

class CreateWalletRequest(BaseModel):
    address:    str
    public_key: str

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 64:
            raise ValueError('Address must be 64 hex characters')
        return v

    @field_validator('public_key')
    @classmethod
    def validate_pubkey(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('public_key is required')
        return v.strip()


class LoginRequest(BaseModel):
    address:    str
    public_key: str
    signature:  str
    nonce:      str

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 64:
            raise ValueError('Address must be 64 hex characters')
        return v

    @field_validator('nonce')
    @classmethod
    def validate_nonce(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('nonce is required')
        return v


# =============================================================================
# Contacts
# =============================================================================

class AddContactRequest(BaseModel):
    address: str
    name:    str

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 64:
            raise ValueError('Address must be 64 hex characters')
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 50:
            raise ValueError('Name must be 1–50 characters')
        return v


class AddContactFromChatRequest(BaseModel):
    contact_address: str
    contact_name:    Optional[str] = ''

    @field_validator('contact_address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 64:
            raise ValueError('Invalid address format (must be 64 hex chars)')
        return v


class DeleteContactRequest(BaseModel):
    address: str

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 64:
            raise ValueError('Invalid address format')
        return v


class EditContactRequest(BaseModel):
    address: str
    name:    str

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) != 64:
            raise ValueError('Address must be 64 hex characters')
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 50:
            raise ValueError('Name must be 1–50 characters')
        return v


# =============================================================================
# Groups
# =============================================================================

class CreateGroupRequest(BaseModel):
    name:    str
    members: List[str]

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 100:
            raise ValueError('Name must be 1–100 characters')
        return v

    @field_validator('members')
    @classmethod
    def validate_members(cls, v: List[str]) -> List[str]:
        if not 1 <= len(v) <= 50:
            raise ValueError('Members must be 1–50')
        return v


class DeleteGroupRequest(BaseModel):
    group_id: str

    @field_validator('group_id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 32:
            raise ValueError('Invalid group ID format')
        return v


class RenameGroupRequest(BaseModel):
    group_id: str
    name:     str

    @field_validator('group_id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 32:
            raise ValueError('Invalid group ID format')
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 100:
            raise ValueError('Name must be 1–100 characters')
        return v


class GroupMemberRequest(BaseModel):
    group_id: str
    address:  str

    @field_validator('group_id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 32:
            raise ValueError('Invalid group ID format')
        return v

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 64:
            raise ValueError('Invalid address (must be 64 hex chars)')
        return v


# =============================================================================
# Messages
# =============================================================================

class SendMessageRequest(BaseModel):
    recipient:      Optional[str] = None
    payload:        Optional[dict] = None
    message_type:   str = 'direct'
    group_id:       Optional[str] = None
    encrypted_map:  Optional[dict] = None
    sender_pubkey: Optional[str] = None

    @field_validator('message_type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ('direct', 'group'):
            raise ValueError('message_type must be direct or group')
        return v

    @field_validator('payload')
    @classmethod
    def validate_payload_size(cls, v):
        if v is not None:
            size = len(json.dumps(v))
            if size > MAX_MESSAGE_PAYLOAD_SIZE:
                raise ValueError(f'Payload too large (max {MAX_MESSAGE_PAYLOAD_SIZE} bytes)')
        return v

    @field_validator('encrypted_map')
    @classmethod
    def validate_encrypted_map_size(cls, v):
        if v is not None:
            size = len(json.dumps(v))
            if size > MAX_MESSAGE_PAYLOAD_SIZE:
                raise ValueError(f'Encrypted map too large (max {MAX_MESSAGE_PAYLOAD_SIZE} bytes)')
        return v


class MarkReadRequest(BaseModel):
    chat_with:       str
    last_message_id: Optional[int] = None


class MessageStatusesRequest(BaseModel):
    ids: List[int] = []


class DeleteMessageRequest(BaseModel):
    message_id: int

    @field_validator('message_id')
    @classmethod
    def validate_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('message_id must be positive')
        return v


class ClearConversationRequest(BaseModel):
    chat_with: str


# =============================================================================
# Wallet
# =============================================================================

class TransferRequest(BaseModel):
    recipient: str
    amount:    int

    @field_validator('recipient')
    @classmethod
    def validate_recipient(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) != 64 or not all(c in '0123456789abcdef' for c in v):
            raise ValueError('Invalid recipient address')
        return v

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class StakeRequest(BaseModel):
    amount: int

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class MineRequest(BaseModel):
    proof:      int
    challenge:  str
    last_proof: int
    last_index: int


# =============================================================================
# Status
# =============================================================================

class HeartbeatRequest(BaseModel):
    current_chat: Optional[str] = ''


class ManyStatusesRequest(BaseModel):
    addresses: List[str] = []

# =============================================================================
# Calls - удаление
# =============================================================================

class DeleteCallRequest(BaseModel):
    call_id: int

    @field_validator('call_id')
    @classmethod
    def validate_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('call_id must be positive')
        return v
# =============================================================================
# Calls (история звонков) – НОВОЕ
# =============================================================================

class CallLogEntry(BaseModel):
    address: str                      # адрес собеседника
    name: Optional[str] = None        # имя собеседника
    direction: str                    # 'incoming' или 'outgoing'
    status: str                       # 'answered', 'missed', 'rejected'
    duration: int = 0                 # длительность в секундах
    timestamp: int                    # Unix timestamp начала звонка