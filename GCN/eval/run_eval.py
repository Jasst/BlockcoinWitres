"""
run_eval.py — CLI для запуска agent_eval harness.

Использование:
    python -m GCN.eval.run_eval [user_id]

Результат:
    - печатает сводку в stdout
    - сохраняет полный отчёт в GCN/eval/reports/<user_id>_<timestamp>.json
"""
import asyncio
import sys
import time
from pathlib import Path

from GCN.ai_assistant import get_assistant
from GCN.eval.agent_eval import create_test_cases, run_agent_eval


async def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else f"eval_{int(time.time())}"
    print(f"Running eval for user: {user_id}")

    assistant = await get_assistant(user_id)

    async def caller(msg: str):
        resp, meta = await assistant.process_input(msg)
        return {"text": resp, "meta": meta or {}}

    def trace_extractor(resp):
        return resp.get("meta", {}).get("tool_trace", [])

    reports_dir = Path("GCN/eval/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f"{user_id}_{int(time.time())}.json"

    try:
        summary = await run_agent_eval(
            caller,
            trace_extractor,
            test_cases=create_test_cases(),
            output_path=output_path,
        )
    finally:
        try:
            await assistant.shutdown()
        except Exception:
            pass

    print()
    print("=== EVAL SUMMARY ===")
    print(f"User:               {user_id}")
    print(f"Total:              {summary.total_tests}")
    print(f"Passed:             {summary.passed}")
    print(f"Failed:             {summary.failed}")
    print(f"Task Success Rate:  {summary.metrics['task_success_rate']:.2%}")
    print(f"Tool Use Accuracy:  {summary.metrics['tool_use_accuracy']:.2%}")
    print(f"Avg Execution Time: {summary.metrics['avg_execution_time']:.2f}s")
    print(f"Report:             {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
