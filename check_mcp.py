import asyncio
import aiohttp

async def check():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://blockcoin.ru/mcp/') as r:
            print(f'Status: {r.status}')
            text = await r.text()
            print(text[:500])

asyncio.run(check())
