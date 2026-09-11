"""
routes/wallet.py — Баланс, переводы, стейкинг, майнинг, глобальная статистика (асинхронная версия)
Использует безопасный анстейкинг с детальным ответом.
"""
import logging
import secrets
import time
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Request

from config import (
    COIN, COIN_NAME, TRANSFER_FEE, MIN_STAKE_AMOUNT, BLOCK_REWARD, CONFIG,
    STAKING_FEE_POOL_ADDRESS, ENABLE_MINING, ENABLE_STAKING, MESSAGE_FEE,
    STAKING_FEE_FROM_BLOCK_REWARD, MAX_SUPPLY, MINING_CHALLENGE_TTL,
)
from dependencies import require_auth, make_rate_limit_dep
from models import TransferRequest, StakeRequest, MineRequest
import services.wallet
from setup import general_limiter
from routes.ws import manager

_mining_challenges = {}
_mining_challenges_lock = Lock()

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/wallet', tags=['wallet'])

_CHALLENGE_TTL = MINING_CHALLENGE_TTL


@router.get('/config')
async def wallet_config(request: Request):
    blockchain = request.app.state.blockchain
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        ratio = await blockchain.get_staking_fee_ratio(conn)
    return {
        'enable_mining':    ENABLE_MINING,
        'enable_staking':   ENABLE_STAKING,
        'message_fee':      MESSAGE_FEE,
        'transfer_fee':     TRANSFER_FEE,
        'block_reward':     BLOCK_REWARD if ENABLE_MINING else 0,
        'coin_name':        COIN_NAME,
        'coin_divisor':     COIN,
        'staking_fee_ratio': ratio,
        'pow_max_iterations': CONFIG['POW_MAX_ITERATIONS'],
        'pow_difficulty':   CONFIG['POW_DIFFICULTY'],
    }


@router.get('/balance')
async def wallet_balance(request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow('SELECT balance FROM wallets WHERE address = $1', address)
        balance = row[0] if row else 0
        stake_row = await conn.fetchrow('SELECT COALESCE(SUM(amount), 0) FROM stakes WHERE address=$1 AND active=1', address)
        staked = stake_row[0] if stake_row and stake_row[0] else 0
    return {
        'address':   address,
        'balance':   balance,
        'staked':    staked,
        'coin':      COIN,
        'coin_name': COIN_NAME,
    }


@router.get('/transactions')
async def wallet_transactions(request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        rows = await conn.fetch('''
            SELECT id, tx_type, sender, recipient, amount, timestamp
            FROM coin_transactions
            WHERE sender = $1 OR recipient = $1
            ORDER BY timestamp DESC LIMIT 50
        ''', address)
        txs = [
            {'id': r[0], 'type': r[1], 'sender': r[2],
             'recipient': r[3], 'amount': r[4], 'timestamp': r[5]}
            for r in rows
        ]
    return {'transactions': txs}


@router.post('/send')
async def wallet_send(body: TransferRequest, request: Request, address: str = Depends(require_auth)):
    if body.recipient == address:
        raise HTTPException(400, 'Cannot send to yourself')
    total = body.amount + TRANSFER_FEE
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        await conn.execute('BEGIN')
        result = await conn.execute(
            'UPDATE wallets SET balance = balance - $1 WHERE address = $2 AND balance >= $1',
            total, address
        )
        if result == "UPDATE 0":
            await conn.execute('ROLLBACK')
            raise HTTPException(400, f'Insufficient balance. Need {total / COIN} {COIN_NAME}')
        await conn.execute(
            'INSERT INTO wallets (address, balance) VALUES ($1, $2) '
            'ON CONFLICT(address) DO UPDATE SET balance = wallets.balance + $2',
            body.recipient, body.amount
        )
        ts = time.time()
        await conn.execute(
            'INSERT INTO coin_transactions (tx_type, sender, recipient, amount, timestamp) '
            'VALUES ($1, $2, $3, $4, $5)',
            'transfer', address, body.recipient, body.amount, ts
        )
        await conn.execute(
            'INSERT INTO wallets (address, balance) VALUES ($1, $2) '
            'ON CONFLICT(address) DO UPDATE SET balance = wallets.balance + $2',
            STAKING_FEE_POOL_ADDRESS, TRANSFER_FEE
        )
        await conn.execute(
            'INSERT INTO coin_transactions (tx_type, sender, recipient, amount, timestamp, note) '
            'VALUES ($1, $2, $3, $4, $5, $6)',
            'fee', address, STAKING_FEE_POOL_ADDRESS, TRANSFER_FEE, ts, 'transfer fee to staking pool'
        )
        if ENABLE_STAKING and services.wallet.staking_manager:
            await services.wallet.staking_manager.add_to_fee_pool(TRANSFER_FEE, cursor=conn)
        await conn.execute('COMMIT')

    # 🔔 Уведомление о переводе
    tx_data = {
        'sender': address,
        'recipient': body.recipient,
        'amount': body.amount,
        'timestamp': ts,
        'tx_type': 'transfer'
    }
    try:
        await manager.send_personal_message(address, {
            'type': 'new_transaction',
            'transaction': tx_data
        })
        if body.recipient != address:
            await manager.send_personal_message(body.recipient, {
                'type': 'new_transaction',
                'transaction': tx_data
            })
    except Exception as e:
        logger.warning(f"Failed to send transfer notification: {e}")

    return {'message': 'Sent', 'amount': body.amount, 'fee': TRANSFER_FEE, 'coin_name': COIN_NAME}


@router.post('/stake')
async def stake(body: StakeRequest, request: Request, address: str = Depends(require_auth)):
    if not ENABLE_STAKING:
        raise HTTPException(403, 'Staking is disabled')
    if services.wallet.staking_manager is None:
        raise HTTPException(503, 'Staking service not initialized')
    if body.amount < MIN_STAKE_AMOUNT:
        raise HTTPException(400, f'Minimum stake is {MIN_STAKE_AMOUNT / COIN:.6f} {COIN_NAME}')
    unlock_block = await services.wallet.staking_manager.stake(address, body.amount)
    if unlock_block == -1:
        raise HTTPException(400, 'Insufficient balance')

    # 🔔 Уведомление о стейкинге
    tx_data = {
        'sender': address,
        'recipient': STAKING_FEE_POOL_ADDRESS,
        'amount': body.amount,
        'timestamp': time.time(),
        'tx_type': 'stake'
    }
    try:
        await manager.send_personal_message(address, {
            'type': 'new_transaction',
            'transaction': tx_data
        })
    except Exception as e:
        logger.warning(f"Failed to send stake notification: {e}")

    return {'message': 'Staked', 'unlock_block': unlock_block}


@router.post('/unstake')
async def unstake(request: Request, address: str = Depends(require_auth)):
    if not ENABLE_STAKING:
        raise HTTPException(403, 'Staking is disabled')
    if services.wallet.staking_manager is None:
        raise HTTPException(503, 'Staking service not initialized')
    result = await services.wallet.staking_manager.unstake(address)
    if not result.get('success'):
        error_msg = result.get('error', 'No active stake or still locked')
        raise HTTPException(400, error_msg)

    # 🔔 Уведомление об анстейкинге
    tx_data = {
        'sender': STAKING_FEE_POOL_ADDRESS,
        'recipient': address,
        'amount': result['total_payout'],
        'timestamp': time.time(),
        'tx_type': 'unstake'
    }
    try:
        await manager.send_personal_message(address, {
            'type': 'new_transaction',
            'transaction': tx_data
        })
    except Exception as e:
        logger.warning(f"Failed to send unstake notification: {e}")

    coin_div = COIN
    return {
        'message': f"Unstaked {result['unstaked_count']} stake(s). Total payout: {result['total_payout'] / coin_div:.6f} {COIN_NAME}",
        'unstaked_count': result['unstaked_count'],
        'total_payout': result['total_payout'],
        'still_locked_count': result['still_locked_count'],
        'failed_due_to_pool': result['failed_due_to_pool'],
        'errors': result['errors'],
        'coin_name': COIN_NAME,
    }


@router.get('/staking/info')
async def staking_info(request: Request, address: str = Depends(require_auth)):
    if not ENABLE_STAKING:
        raise HTTPException(403, 'Staking is disabled')
    if services.wallet.staking_manager is None:
        raise HTTPException(503, 'Staking service not initialized')
    blockchain = request.app.state.blockchain
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        rows = await conn.fetch(
            'SELECT amount, start_time, start_block, unlock_block '
            'FROM stakes WHERE address=$1 AND active=1',
            address
        )
        stakes = [dict(r) for r in rows]
        last_block = await blockchain._last_block_raw(conn)
        current_block = last_block.get('block_index', 0)
    expected_income = await services.wallet.staking_manager.get_expected_income(address)
    return {
        'stakes': stakes,
        'expected_income': expected_income,
        'current_block': current_block,
        'coin_name': COIN_NAME,
        'coin_divisor': COIN,
    }


@router.get('/last-proof', dependencies=[Depends(make_rate_limit_dep(general_limiter, limit=10))])
async def last_proof(request: Request, address: str = Depends(require_auth)):
    blockchain = request.app.state.blockchain
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        last = await blockchain._last_block_raw(conn)
        difficulty = await blockchain.get_difficulty(conn)
    last_index = last.get('block_index', 0)
    challenge = secrets.token_hex(16)
    with _mining_challenges_lock:
        _mining_challenges.setdefault(address, {})[challenge] = time.time() + _CHALLENGE_TTL
        MAX_CHALLENGES_TOTAL = 1000
        if len(_mining_challenges) > MAX_CHALLENGES_TOTAL:
            oldest_addr = min(_mining_challenges.items(), key=lambda kv: min(kv[1].values()))[0]
            del _mining_challenges[oldest_addr]
    return {
        'last_proof': last.get('proof', 0),
        'last_index': last_index,
        'difficulty': difficulty,
        'challenge': challenge,
    }


@router.post('/mine', dependencies=[Depends(make_rate_limit_dep(general_limiter, limit=10))])
async def mine(body: MineRequest, request: Request, address: str = Depends(require_auth)):
    if not ENABLE_MINING:
        raise HTTPException(403, 'Mining disabled')
    blockchain = request.app.state.blockchain
    with _mining_challenges_lock:
        challenges = _mining_challenges.get(address, {})
        if body.challenge not in challenges or time.time() > challenges[body.challenge]:
            logger.warning(f"Mining challenge expired for {address[:16]}...")
            raise HTTPException(400, 'Invalid or expired challenge')
        del _mining_challenges[address][body.challenge]
        if not _mining_challenges[address]:
            del _mining_challenges[address]
    success, error_msg, reward_amount, block_index = await blockchain.try_mine_block(
        body.last_proof, body.last_index, body.proof, body.challenge, address
    )
    if not success:
        logger.warning(f"Mining failed for {address}: {error_msg}")
        status_code = 409 if error_msg == 'Blockchain moved, try again' else 400
        raise HTTPException(status_code, error_msg)
    logger.info(f"Block {block_index} mined by {address}, reward: {reward_amount}")

    # 🔔 Уведомление о награде за блок
    tx_data = {
        'sender': 'blockchain',
        'recipient': address,
        'amount': reward_amount,
        'timestamp': time.time(),
        'tx_type': 'block_reward'
    }
    try:
        await manager.send_personal_message(address, {
            'type': 'new_transaction',
            'transaction': tx_data
        })
    except Exception as e:
        logger.warning(f"Failed to send mining reward notification: {e}")

    await manager.broadcast({
        'type': 'new_block',
        'last_proof': body.proof,
        'last_index': block_index,
    })

    return {
        'message': 'Block mined',
        'reward': reward_amount,
        'block_index': block_index,
        'coin_name': COIN_NAME,
    }


@router.get('/global-stats')
async def wallet_global_stats(request: Request):
    from database import get_db_cursor
    blockchain = request.app.state.blockchain
    async with get_db_cursor() as conn:
        ratio = await blockchain.get_staking_fee_ratio(conn)
        total_supply_raw = (await conn.fetchval('SELECT COALESCE(SUM(balance), 0) FROM wallets')) or 0
        row = await conn.fetchrow('SELECT balance FROM wallets WHERE address = $1', STAKING_FEE_POOL_ADDRESS)
        staking_pool_balance = row[0] if row else 0
        total_blocks = (await conn.fetchval('SELECT COUNT(*) FROM blockchain')) or 0
        total_staked_raw = (await conn.fetchval('SELECT COALESCE(SUM(amount), 0) FROM stakes WHERE active = 1')) or 0

        difficulty = await blockchain.get_difficulty(conn)

    max_supply_sats = MAX_SUPPLY * COIN if MAX_SUPPLY else None
    if max_supply_sats is not None:
        remaining_sats = max(0, max_supply_sats - total_supply_raw)
    else:
        remaining_sats = None

    return {
        'total_supply': total_supply_raw,
        'staking_pool_balance': staking_pool_balance,
        'block_reward': BLOCK_REWARD if ENABLE_MINING else 0,
        'total_blocks': total_blocks,
        'total_staked': total_staked_raw,
        'difficulty': difficulty,
        'coin_name': COIN_NAME,
        'coin_divisor': COIN,
        'max_supply': MAX_SUPPLY,
        'staking_fee_ratio': ratio,
        'remaining_supply': remaining_sats,
        'message_fee': MESSAGE_FEE,
    }