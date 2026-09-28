#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thought_loop.py
===============

Экспериментальная система для исследования пошаговой генерации LLM.

Режимы:
    single      Один обычный вызов.
    raw         Ответ по кускам; между запросами ничего нового в контекст.
    reflective  После каждого куска маркер дописывается В КОНЕЦ последнего
                assistant-сообщения (см. ниже).
    critique    После каждого куска выполняется ОТДЕЛЬНЫЙ вызов критика; его
                вердикт подмешивается в контекст так же, как маркер в
                reflective — внутрь assistant-сообщения, чтобы модель
                ПРОДОЛЖАЛА, а не начинала заново.
    chat        Интерактивный чат с теми же механизмами.

ПОЧЕМУ МАРКЕР ВНУТРИ ASSISTANT, А НЕ ОТДЕЛЬНЫМ USER-СООБЩЕНИЕМ
----------------------------------------------------------------
Chat-модели обучены: последнее сообщение role=user → нужно ответить
с нуля. Если после assistant-сообщения положить user-сообщение с
маркером «продолжи», модель воспринимает это как новый вопрос и
начинает ответ заново (мы это видели в первом эксперименте — цикл
перезапуска «Энтропия — это ...»).

Обходной путь: дописать маркер прямо в конец assistant-сообщения.
Тогда контекст по-прежнему заканчивается на assistant, и модель
генерирует следующее assistant-сообщение как ПРОДОЛЖЕНИЕ.

Это работает не со всеми бэкендами одинаково хорошо, но на
инструкт-моделях (Qwen, Llama-3-Instruct, Mistral-Instruct) даёт
заметно более осмысленный результат, чем маркер как user.

ВАЖНО
-----
Этот эксперимент НЕ доказывает наличие/отсутствие сознания,
"внутреннего мышления" или AGI.

Reasoning-модели (Qwen3-thinking, DeepSeek-R1, QwQ, R1-distill и т.п.)
кладут свои токены в скрытую фазу "thinking". Если она не закончилась до
исчерпания max_tokens — content приходит пустым, finish_reason=length.

Зависимости: pip install requests
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


DEFAULT_URL = "http://localhost:1234/v1/chat/completions"

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ============================================================================
# Данные эксперимента
# ============================================================================

@dataclass
class GenerationStats:
    requests: int = 0
    generated_chars: int = 0
    generated_pieces: int = 0
    logprob_sum: float = 0.0
    logprob_count: int = 0
    elapsed: float = 0.0
    finish_reason: Optional[str] = None
    usage_prompt_tokens: int = 0
    usage_completion_tokens: int = 0
    usage_total_tokens: int = 0
    context_chars_last: int = 0
    errors: int = 0

    def add_logprob(self, value: float) -> None:
        self.logprob_sum += value
        self.logprob_count += 1

    @property
    def mean_logprob(self) -> Optional[float]:
        if self.logprob_count == 0:
            return None
        return self.logprob_sum / self.logprob_count

    @property
    def perplexity(self) -> Optional[float]:
        if self.mean_logprob is None:
            return None
        try:
            return math.exp(-self.mean_logprob)
        except OverflowError:
            return float("inf")


@dataclass
class GenerationResult:
    text: str
    stats: GenerationStats = field(default_factory=GenerationStats)
    step_texts: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    raw_responses: list[dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Токенизация и сравнение
# ============================================================================

def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence. O(len(a) * len(b))."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for ai in a:
        current = [0] * (len(b) + 1)
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def compare_outputs(a: str, b: str) -> dict[str, Any]:
    ta, tb = tokens(a), tokens(b)
    sa, sb = set(ta), set(tb)
    union = sa | sb
    jaccard = (len(sa & sb) / len(union)) if union else 1.0
    lcs = lcs_length(ta, tb)
    denom = max(len(ta), len(tb), 1)
    return {
        "exact_match": (a or "").strip() == (b or "").strip(),
        "a_tokens": len(ta),
        "b_tokens": len(tb),
        "jaccard": round(jaccard, 4),
        "lcs_ratio": round(lcs / denom, 4),
        "len_ratio": round((len(tb) / len(ta)) if ta else 0.0, 4),
        "char_ratio": round((len(b) / len(a)) if a else 0.0, 4),
    }


# ============================================================================
# Разбор ответа OpenAI-совместимого API
# ============================================================================

def extract_message(data: dict[str, Any], debug: bool = False) -> tuple[str, str]:
    """Возвращает (content, reasoning)."""
    choices = data.get("choices") or []
    if not choices:
        return "", ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

    # Некоторые бэкенды возвращают content как список блоков.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        content = "".join(parts)

    if not isinstance(content, str):
        content = str(content)
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    if debug and not content:
        keys = sorted(message.keys()) if isinstance(message, dict) else []
        print(f"[extract_message debug] message keys = {keys}")
        print(f"[extract_message debug] reasoning_len = {len(reasoning)}")

    return content, reasoning


def extract_logprobs(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices") or []
    if not choices:
        return []
    choice = choices[0] or {}
    lp = choice.get("logprobs")
    if not lp:
        return []
    content = lp.get("content")
    if not isinstance(content, list):
        return []
    return content


def extract_usage(data: dict[str, Any]) -> tuple[int, int, int]:
    usage = data.get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("total_tokens") or 0),
    )


def extract_finish_reason(data: dict[str, Any]) -> Optional[str]:
    choices = data.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


# ============================================================================
# Один HTTP-вызов
# ============================================================================

def call_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    seed: Optional[int] = None,
    show_logprobs: bool = False,
    reasoning_effort: Optional[str] = None,
    debug: bool = False,
) -> tuple[str, str, Optional[str], list[dict[str, Any]], dict[str, Any], float]:
    """Возвращает (content, reasoning, finish_reason, logprobs, raw_data, elapsed)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if show_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    started = time.perf_counter()
    response = requests.post(base_url, headers=headers, json=payload, timeout=300)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()

    if debug:
        print("\n========== RAW API RESPONSE ==========")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("======================================\n")

    content, reasoning = extract_message(data, debug=debug)
    finish_reason = extract_finish_reason(data)
    logprobs = extract_logprobs(data)
    return content, reasoning, finish_reason, logprobs, data, elapsed


# ============================================================================
# Сборка messages
# ============================================================================
#
# ВАЖНО: маркер (reflective) и критика (critique) идут ВНУТРЬ последнего
# assistant-сообщения, а не отдельным user-сообщением. Иначе chat-модель
# видит контекст, заканчивающийся на user, и генерирует новый ответ с нуля
# (мы это наблюдали в первом эксперименте: цикл «Энтропия — это…»).

def initial_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    result = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    result.append({"role": "user", "content": user_prompt})
    return result


def messages_raw(system_prompt: str, prompt: str, previous: str) -> list[dict[str, str]]:
    messages = initial_messages(system_prompt, prompt)
    if previous:
        messages.append({"role": "assistant", "content": previous})
    return messages


def messages_reflective(system_prompt: str, prompt: str, previous: str,
                        marker: str) -> list[dict[str, str]]:
    messages = initial_messages(system_prompt, prompt)
    if previous:
        # Маркер дописан в конец assistant-сообщения: контекст остаётся
        # "assistant …", и следующий шаг — продолжение, а не новый ответ.
        messages.append({
            "role": "assistant",
            "content": f"{previous}\n\n{marker}",
        })
    return messages


def messages_continue_after_critique(system_prompt: str, prompt: str,
                                     previous: str, critique: str,
                                     instruction: str) -> list[dict[str, str]]:
    messages = initial_messages(system_prompt, prompt)
    if previous:
        # Критика + инструкция идут в конец assistant-сообщения — тем же
        # принципом, что и маркер в reflective. Внутри сообщения она для
        # модели выглядит как её собственная заметка к самой себе.
        messages.append({
            "role": "assistant",
            "content": (
                f"{previous}\n\n"
                f"[{instruction}]\n"
                f"[Критика: {critique}]\n"
                f"[Продолжи с учётом этого:]"
            ),
        })
    return messages


def messages_for_critic(system_prompt: str, prompt: str, previous: str,
                        critic_prompt: str) -> list[dict[str, str]]:
    # Критик — это отдельный "ролевой" вызов: он отвечает на НОВЫЙ вопрос
    # о предыдущем тексте, а не продолжает его. Здесь user-сообщение уместно.
    messages = initial_messages(system_prompt, prompt)
    messages.append({"role": "assistant", "content": previous})
    messages.append({"role": "user", "content": critic_prompt})
    return messages


# ============================================================================
# Обновление статистики
# ============================================================================

def update_stats(stats: GenerationStats, data: dict[str, Any],
                 logprobs: list[dict[str, Any]], elapsed: float,
                 text: str, context_chars: int) -> None:
    stats.requests += 1
    stats.generated_pieces += 1
    stats.generated_chars += len(text)
    stats.elapsed += elapsed
    stats.context_chars_last = context_chars

    prompt_tokens, completion_tokens, total_tokens = extract_usage(data)
    stats.usage_prompt_tokens += prompt_tokens
    stats.usage_completion_tokens += completion_tokens
    stats.usage_total_tokens += total_tokens

    stats.finish_reason = extract_finish_reason(data)

    for item in logprobs:
        value = item.get("logprob")
        if value is None:
            continue
        try:
            stats.add_logprob(float(value))
        except (TypeError, ValueError):
            pass


def warn_empty_response(finish_reason: Optional[str], completion_tokens: int,
                        context_label: str) -> None:
    """Явно объясняет пустой ответ — чтобы пользователь не гадал."""
    print(
        f"\n[{context_label}] ПУСТО: content=\"\", "
        f"finish_reason={finish_reason}, completion_tokens={completion_tokens}"
    )
    if finish_reason == "length" and completion_tokens > 0:
        print(
            "  → Похоже, модель израсходовала весь max_tokens на скрытую "
            "фазу reasoning (Qwen3-thinking / DeepSeek-R1 / QwQ).\n"
            "  → Решения: (а) возьмите не-reasoning модель; "
            "(б) добавьте '/no_think' к промпту; "
            "(в) передайте --reasoning-effort none; "
            "(г) увеличьте --chunk и --max-steps."
        )
    elif completion_tokens == 0:
        print("  → Модель не сгенерировала ни одного токена.")


# ============================================================================
# Детектор циклов
# ============================================================================

def repeated_piece(pieces: list[str], count: int = 4) -> bool:
    if len(pieces) < count:
        return False
    recent = pieces[-count:]
    return len(set(recent)) == 1 and bool(recent[0])


def repeated_suffix(text: str, repetitions: int = 3, min_chars: int = 10) -> bool:
    if len(text) < min_chars * repetitions:
        return False
    for size in range(min_chars, min(200, len(text) // repetitions) + 1):
        suffix = text[-size:]
        if text.endswith(suffix * repetitions):
            return True
    return False


# ============================================================================
# Single
# ============================================================================

def run_single(args) -> GenerationResult:
    result = GenerationResult(text="")
    messages = initial_messages(args.system, args.prompt)

    try:
        (text, reasoning, finish_reason, logprobs, data, elapsed) = call_api(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_steps * args.chunk,
            seed=args.seed,
            show_logprobs=args.show_logprobs,
            reasoning_effort=args.reasoning_effort,
            debug=args.debug,
        )
    except Exception as exc:
        result.stats.errors += 1
        print(f"[single error] {exc}")
        return result

    result.text = text
    result.step_texts.append(text)
    update_stats(
        result.stats, data, logprobs, elapsed, text,
        context_chars=sum(len(m.get("content", "")) for m in messages),
    )

    if reasoning and args.show_reasoning:
        print("\n[reasoning]\n" + reasoning)

    if text:
        print(text)
    else:
        _, completion_tokens, _ = extract_usage(data)
        warn_empty_response(finish_reason, completion_tokens, "single")

    print(f"\n[finish_reason={finish_reason}]")
    return result


# ============================================================================
# RAW / REFLECTIVE
# ============================================================================

def run_loop(args, mode: str) -> GenerationResult:
    result = GenerationResult(text="")
    so_far = ""
    pieces = []

    print(f"\n=== {mode.upper()} ===\n"
          f"chunk={args.chunk} | max_steps={args.max_steps} | T={args.temperature}\n")

    for step in range(1, args.max_steps + 1):
        if mode == "raw":
            messages = messages_raw(args.system, args.prompt, so_far)
        elif mode == "reflective":
            messages = messages_reflective(args.system, args.prompt, so_far, args.marker)
        else:
            raise ValueError(f"Unknown loop mode: {mode}")

        context_chars = sum(len(m.get("content", "")) for m in messages)

        try:
            (piece, reasoning, finish_reason, logprobs, data, elapsed) = call_api(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.chunk,
                seed=args.seed,
                show_logprobs=args.show_logprobs,
                reasoning_effort=args.reasoning_effort,
                debug=args.debug,
            )
        except Exception as exc:
            result.stats.errors += 1
            print(f"\n[step {step} error] {exc}")
            break

        update_stats(result.stats, data, logprobs, elapsed, piece, context_chars)

        if not piece:
            _, completion_tokens, _ = extract_usage(data)
            warn_empty_response(finish_reason, completion_tokens, f"{mode} step {step}")
            break

        so_far += piece
        pieces.append(piece)
        result.step_texts.append(piece)
        print(f"[{step:03d}] {piece!r}")

        if args.show_logprobs and logprobs:
            first = logprobs[0]
            top = first.get("top_logprobs")
            if top:
                print("      top:", [
                    (x.get("token"), round(float(x.get("logprob", 0)), 3))
                    for x in top
                ])

        if reasoning and args.show_reasoning:
            print("\n      [reasoning]")
            print(reasoning)

        if repeated_piece(pieces):
            print("\n[cycle detected: same piece repeated]")
            break
        if repeated_suffix(so_far):
            print("\n[cycle detected: repeated suffix]")
            break
        if finish_reason == "stop":
            print("\n[model stopped]")
            break

    result.text = so_far
    print("\n=== FINAL ===")
    print(so_far)
    return result


# ============================================================================
# CRITIQUE (внешний критик + продолжение с учётом критики)
# ============================================================================

def run_critique(args) -> GenerationResult:
    result = GenerationResult(text="")
    so_far = ""
    pieces = []

    print("\n=== CRITIQUE / SELF-REFINE ===\n")

    for step in range(1, args.max_steps + 1):
        # Первый шаг — обычная генерация.
        if not so_far:
            messages = initial_messages(args.system, args.prompt)
            context_chars = sum(len(m.get("content", "")) for m in messages)

            try:
                (piece, reasoning, finish_reason, logprobs, data, elapsed) = call_api(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.chunk,
                    seed=args.seed,
                    show_logprobs=args.show_logprobs,
                    reasoning_effort=args.reasoning_effort,
                    debug=args.debug,
                )
            except Exception as exc:
                result.stats.errors += 1
                print(f"[initial generation error] {exc}")
                break

            update_stats(result.stats, data, logprobs, elapsed, piece, context_chars)

            if not piece:
                _, completion_tokens, _ = extract_usage(data)
                warn_empty_response(finish_reason, completion_tokens, "critique step 1")
                break

            so_far += piece
            pieces.append(piece)
            print(f"[{step:03d}] {piece!r}")

            if finish_reason == "stop":
                break
            continue

        # Отдельный критик. Критика НЕ становится частью so_far.
        critic_messages = messages_for_critic(
            args.system, args.prompt, so_far, args.critic_prompt,
        )
        try:
            (critique, _critic_reasoning, _critic_finish,
             _critic_logprobs, _critic_data, _critic_elapsed) = call_api(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                messages=critic_messages,
                temperature=args.critic_temperature,
                max_tokens=args.critic_max_tokens,
                seed=(args.seed + 100000 + step) if args.seed is not None else None,
                show_logprobs=False,
                reasoning_effort=args.reasoning_effort,
                debug=args.debug,
            )
        except Exception as exc:
            result.stats.errors += 1
            print(f"\n[critic error] {exc}")
            break

        result.critiques.append(critique.strip())
        print(f"\n      [CRITIC]\n      {critique.strip()}\n")

        # Продолжение с учётом критики.
        # ИСПРАВЛЕНО: критика встраивается внутрь assistant-сообщения
        # (см. messages_continue_after_critique), а не как новый user-turn.
        continuation_messages = messages_continue_after_critique(
            args.system, args.prompt, so_far, critique, args.critic_instruction,
        )
        context_chars = sum(len(m.get("content", "")) for m in continuation_messages)

        try:
            (piece, reasoning, finish_reason, logprobs, data, elapsed) = call_api(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                messages=continuation_messages,
                temperature=args.temperature,
                max_tokens=args.chunk,
                seed=args.seed,
                show_logprobs=args.show_logprobs,
                reasoning_effort=args.reasoning_effort,
                debug=args.debug,
            )
        except Exception as exc:
            result.stats.errors += 1
            print(f"\n[continuation error] {exc}")
            break

        update_stats(result.stats, data, logprobs, elapsed, piece, context_chars)

        if not piece:
            _, completion_tokens, _ = extract_usage(data)
            warn_empty_response(finish_reason, completion_tokens, f"critique step {step}")
            break

        so_far += piece
        pieces.append(piece)
        result.step_texts.append(piece)
        print(f"[{step:03d}] {piece!r}")

        if repeated_piece(pieces):
            print("\n[cycle detected]")
            break
        if repeated_suffix(so_far):
            print("\n[repeated suffix detected]")
            break
        if finish_reason == "stop":
            print("\n[model stopped]")
            break

    result.text = so_far
    print("\n=== FINAL CRITIQUE RESULT ===")
    print(so_far)
    return result


# ============================================================================
# Печать статистики
# ============================================================================

def print_stats(name: str, result: GenerationResult) -> None:
    s = result.stats
    print(f"\n[{name}]")
    print(f"  chars              = {len(result.text)}")
    print(f"  requests           = {s.requests}")
    print(f"  elapsed            = {s.elapsed:.2f}s")
    if s.elapsed > 0:
        print(f"  chars/sec          = {len(result.text) / s.elapsed:.2f}")
    print(f"  prompt_tokens      = {s.usage_prompt_tokens}")
    print(f"  completion_tokens  = {s.usage_completion_tokens}")
    print(f"  total_tokens       = {s.usage_total_tokens}")
    print(f"  finish_reason      = {s.finish_reason}")
    if s.mean_logprob is not None:
        print(f"  mean_logprob       = {s.mean_logprob:.4f}")
        print(f"  approx_perplexity  = {s.perplexity:.3f}")
    print(f"  last_context_chars = {s.context_chars_last}")
    print(f"  errors             = {s.errors}")


def print_pairwise(results: dict[str, GenerationResult]) -> None:
    names = list(results)
    print("\n\n" + "=" * 60)
    print("PAIRWISE COMPARISON")
    print("=" * 60)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            metrics = compare_outputs(results[a_name].text, results[b_name].text)
            print(f"\n{a_name}  vs  {b_name}")
            for key, value in metrics.items():
                print(f"  {key:16s} = {value}")


def print_experiment_summary(results: dict[str, GenerationResult]) -> None:
    print("\n\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print_stats(name, result)
    print_pairwise(results)


# ============================================================================
# Агрегация по нескольким прогонам
# ============================================================================

def aggregate_runs(all_runs: dict[str, list[GenerationResult]]) -> None:
    print("\n\n" + "=" * 60)
    print("MULTI-RUN STATISTICS")
    print("=" * 60)
    for name, runs in all_runs.items():
        lengths = [len(x.text) for x in runs]
        elapsed = [x.stats.elapsed for x in runs]
        logprobs = [x.stats.mean_logprob for x in runs if x.stats.mean_logprob is not None]

        print(f"\n{name}")
        if lengths:
            print(f"  length mean = {statistics.mean(lengths):.2f}")
            if len(lengths) > 1:
                print(f"  length stdev = {statistics.stdev(lengths):.2f}")
        if elapsed:
            print(f"  time mean = {statistics.mean(elapsed):.2f}s")
        if logprobs:
            print(f"  logprob mean = {statistics.mean(logprobs):.4f}")


# ============================================================================
# Chat
# ============================================================================

def run_chat(args) -> None:
    history: list[dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    print("\n" + "=" * 60)
    print("THOUGHT LOOP CHAT")
    print("=" * 60)
    print(f"style={args.chat_style}\nchunk={args.chunk}\nmax_steps={args.max_steps}")
    print("\nexit / quit / пустая строка — выход.\n")

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "выход"):
            break

        history.append({"role": "user", "content": user_input})

        so_far = ""
        pieces = []
        print("\nМодель:")

        for step in range(1, args.max_steps + 1):
            # Собираем контекст для текущего шага: история + накопленный so_far.
            # ИСПРАВЛЕНО: маркер/критика встраиваются внутрь assistant-сообщения
            # с so_far (см. комментарий к messages_reflective).
            messages = list(history)

            if so_far:
                if args.chat_style == "reflective":
                    messages.append({
                        "role": "assistant",
                        "content": f"{so_far}\n\n{args.marker}",
                    })

                elif args.chat_style == "critique":
                    # Отдельный critic-вызов — это "новый вопрос о so_far",
                    # здесь user-сообщение уместно.
                    critic_messages = list(history)
                    critic_messages.append({"role": "assistant", "content": so_far})
                    critic_messages.append({"role": "user", "content": args.critic_prompt})

                    try:
                        (critique, _, _, _, _, _) = call_api(
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            messages=critic_messages,
                            temperature=args.critic_temperature,
                            max_tokens=args.critic_max_tokens,
                            seed=None,
                            show_logprobs=False,
                            reasoning_effort=args.reasoning_effort,
                            debug=args.debug,
                        )
                    except Exception as exc:
                        print(f"\n[critic error] {exc}")
                        break

                    print(f"\n  [critic] {critique.strip()}")
                    messages.append({
                        "role": "assistant",
                        "content": (
                            f"{so_far}\n\n"
                            f"[{args.critic_instruction}]\n"
                            f"[Критика: {critique}]\n"
                            f"[Продолжи с учётом этого:]"
                        ),
                    })

                else:
                    # raw — просто продолжаем assistant-сообщение.
                    messages.append({"role": "assistant", "content": so_far})

            try:
                (piece, reasoning, finish_reason, logprobs, _, _) = call_api(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.chunk,
                    seed=args.seed,
                    show_logprobs=args.show_logprobs,
                    reasoning_effort=args.reasoning_effort,
                    debug=args.debug,
                )
            except Exception as exc:
                print(f"\n[error] {exc}")
                break

            if not piece:
                break

            so_far += piece
            pieces.append(piece)
            print(f"[{step:03d}] {piece!r}")

            if args.show_logprobs and logprobs:
                first = logprobs[0]
                print("      logprob=", first.get("logprob"))

            if reasoning and args.show_reasoning:
                print("\n[reasoning]\n" + reasoning)

            if repeated_piece(pieces):
                print("\n[cycle detected]")
                break
            if repeated_suffix(so_far):
                print("\n[repeated suffix detected]")
                break
            if finish_reason == "stop":
                break

        print("\nМодель:", so_far)

        if so_far:
            history.append({"role": "assistant", "content": so_far})
        else:
            history.pop()  # ничего не сгенерировано — убираем user-сообщение


# ============================================================================
# Compare
# ============================================================================

def run_compare(args) -> None:
    modes = ["single", "raw", "reflective"]
    if args.include_critique:
        modes.append("critique")

    # ---- Один эксперимент ----
    if args.runs <= 1:
        results: dict[str, GenerationResult] = {}
        for mode in modes:
            print("\n\n" + "#" * 60)
            print(f"\n### {mode.upper()}")

            if mode == "single":
                results[mode.upper()] = run_single(args)
            elif mode == "critique":
                results[mode.upper()] = run_critique(args)
            else:
                results[mode.upper()] = run_loop(args, mode)

        print_experiment_summary(results)
        return

    # ---- Несколько seed ----
    all_runs: dict[str, list[GenerationResult]] = {mode: [] for mode in modes}
    base_seed = args.seed

    for run_index in range(args.runs):
        run_seed = None if base_seed is None else base_seed + run_index
        args.seed = run_seed

        print("\n\n" + "#" * 60)
        print(f"\nRUN {run_index + 1}/{args.runs} seed={run_seed}")

        for mode in modes:
            print(f"\n### {mode.upper()}")
            if mode == "single":
                result = run_single(args)
            elif mode == "critique":
                result = run_critique(args)
            else:
                result = run_loop(args, mode)
            all_runs[mode].append(result)

    args.seed = base_seed

    aggregate_runs(all_runs)

    # ---- Межрежимная похожесть ----
    print("\n\n" + "=" * 60)
    print("CROSS-MODE SIMILARITY")
    print("=" * 60)
    for i, a_name in enumerate(modes):
        for b_name in modes[i + 1:]:
            values = []
            for a in all_runs[a_name]:
                for b in all_runs[b_name]:
                    values.append(compare_outputs(a.text, b.text))
            if not values:
                continue
            print(f"\n{a_name.upper()} vs {b_name.upper()}")
            for key in ("jaccard", "lcs_ratio", "len_ratio", "char_ratio"):
                nums = [x[key] for x in values]
                print(f"  {key:16s} = {statistics.mean(nums):.4f}")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Экспериментальная система пошаговой генерации LLM",
    )

    parser.add_argument("--base-url", default=DEFAULT_URL,
                        help="OpenAI-compatible /chat/completions URL")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--system", default="")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--mode",
                        choices=["single", "raw", "reflective", "critique", "chat"],
                        default="raw")
    parser.add_argument("--chat-style",
                        choices=["raw", "reflective", "critique"],
                        default="raw")
    parser.add_argument("--marker",
                        default="[Продолжи свой ответ с того места, где остановился. Не начинай заново.]")
    parser.add_argument("--critic-prompt",
                        default=(
                            "Проанализируй предыдущий фрагмент ответа относительно "
                            "исходного вопроса. Укажи только: "
                            "(1) отклонение от задачи; (2) фактические ошибки; "
                            "(3) важные пропуски. Не переписывай ответ."
                        ))
    parser.add_argument("--critic-instruction",
                        default=(
                            "Используй приведённую критику как корректирующую "
                            "информацию. Не цитируй и не повторяй критику. "
                            "Продолжи исходный ответ естественно."
                        ))
    parser.add_argument("--critic-temperature", type=float, default=0.0)
    parser.add_argument("--critic-max-tokens", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1,
                        help="Количество токенов за один generation request")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--show-logprobs", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Например none/low/medium/high. Зависит от backend.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--include-critique", action="store_true")

    args = parser.parse_args()

    if args.mode != "chat" and not args.prompt:
        parser.error("--prompt обязателен для выбранного режима")

    if args.mode == "chat":
        run_chat(args)
        return 0

    if args.compare:
        run_compare(args)
        return 0

    if args.mode == "single":
        result = run_single(args)
    elif args.mode == "critique":
        result = run_critique(args)
    else:
        result = run_loop(args, args.mode)

    print_stats(args.mode.upper(), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())