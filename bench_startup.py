import asyncio
import time

USER = "2a0fe15382ab71ba6a0602049052dd58f0110d73e1d6947d207a816b3365ab8a"

async def main():
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Старт")

    import GCN.config_ai
    print(f"[{time.time()-t0:6.1f}s] config_ai imported")

    from GCN.memory_service import MemoryService
    print(f"[{time.time()-t0:6.1f}s] memory_service module imported")

    t_ms = time.time()
    ms = MemoryService(USER)
    print(f"[{time.time()-t0:6.1f}s] MemoryService init ({time.time()-t_ms:.1f}s)")

    from routes.ai_assistant import CognitiveController
    t_cc = time.time()
    ctl = CognitiveController(USER)
    print(f"[{time.time()-t0:6.1f}s] CognitiveController init ({time.time()-t_cc:.1f}s)")
    print(f"\nИТОГО: {time.time()-t0:.1f}s")

asyncio.run(main())