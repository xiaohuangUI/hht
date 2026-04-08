from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests


@dataclass(frozen=True)
class LLMSettings:
    api_base: str
    api_key: str
    model: str
    timeout: int = 45


def is_llm_configured(settings: LLMSettings) -> bool:
    return bool(settings.api_base.strip() and settings.api_key.strip() and settings.model.strip())


def _post_chat(
    settings: LLMSettings,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> Tuple[bool, str]:
    if not is_llm_configured(settings):
        return False, "大模型配置不完整，请填写 API Base / API Key / Model。"

    endpoint = settings.api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=settings.timeout)
        if resp.status_code >= 400:
            return False, f"接口请求失败({resp.status_code})：{resp.text[:300]}"

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return False, "模型返回为空。"
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not content:
            return False, "模型未返回有效文本。"
        return True, content
    except Exception as e:
        return False, f"连接异常：{e}"


def chat_with_llm(
    settings: LLMSettings,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> Tuple[bool, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _post_chat(settings, messages, temperature=temperature, max_tokens=max_tokens)


def test_llm_connection(settings: LLMSettings) -> Tuple[bool, str]:
    return _post_chat(
        settings=settings,
        messages=[
            {"role": "system", "content": "你是连接测试助手，请只回复“连接成功”。"},
            {"role": "user", "content": "请回复连接成功"},
        ],
        temperature=0.0,
        max_tokens=30,
    )


def enhance_answer_with_llm(
    settings: LLMSettings,
    user_question: str,
    deterministic_answer: str,
    context_summary: str,
) -> Tuple[bool, str]:
    return _post_chat(
        settings=settings,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是城市交通治理答辩助手。"
                    "请基于给定上下文和规则答案，生成简洁、专业、可展示的中文回复。"
                    "禁止编造数据，保留关键数字，控制在220字内。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{user_question}\n\n"
                    f"规则答案：{deterministic_answer}\n\n"
                    f"上下文：{context_summary}\n\n"
                    "请给出最终回答。"
                ),
            },
        ],
        temperature=0.25,
        max_tokens=360,
    )

