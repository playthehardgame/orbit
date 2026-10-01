from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

from .qwen_route_prefix import (
    QwenRoutePrefixConfig,
    QwenRoutePrefixSpec,
    QwenRoutePrefixStatus,
    derive_qwen_route_prefix_spec,
)


MINICPM5_ROUTE_PREFIX_ENV = "ORBIT_MINICPM5_ROUTE_PREFIX_REUSE"
MINICPM5_ROUTE_PREFIX_TOKEN_COUNT = 768
MINICPM5_ROUTE_PREFIX_FORMAT_VERSION = "minicpm5-route-prefix-v1"
MINICPM5_ROUTE_TOKENIZER_IDENTITY = "gpt2:minicpm5"


def resolve_minicpm5_route_prefix_reuse(
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = True,
) -> QwenRoutePrefixConfig:
    env = os.environ if environ is None else environ
    configured = env.get(MINICPM5_ROUTE_PREFIX_ENV)
    if configured is None:
        return QwenRoutePrefixConfig(enabled=default_enabled, source="default")
    value = configured.strip()
    if value == "1":
        return QwenRoutePrefixConfig(enabled=True, source="stable")
    if value == "0":
        return QwenRoutePrefixConfig(enabled=False, source="stable")
    return QwenRoutePrefixConfig(
        enabled=False,
        source="stable",
        validation_error="invalid_minicpm5_route_prefix_reuse_value",
    )


def derive_minicpm5_route_prefix_spec(
    *,
    system_prompt: str,
    full_prompt: str,
    full_tokens: Sequence[int],
    render_reference: Callable[[str], str],
    tokenize: Callable[[str], list[int]],
) -> tuple[QwenRoutePrefixSpec | None, str | None]:
    return derive_qwen_route_prefix_spec(
        system_prompt=system_prompt,
        full_prompt=full_prompt,
        full_tokens=full_tokens,
        render_reference=render_reference,
        tokenize=tokenize,
        prefix_token_count=MINICPM5_ROUTE_PREFIX_TOKEN_COUNT,
    )


__all__ = [
    "MINICPM5_ROUTE_PREFIX_ENV",
    "MINICPM5_ROUTE_PREFIX_FORMAT_VERSION",
    "MINICPM5_ROUTE_PREFIX_TOKEN_COUNT",
    "MINICPM5_ROUTE_TOKENIZER_IDENTITY",
    "QwenRoutePrefixSpec",
    "QwenRoutePrefixStatus",
    "derive_minicpm5_route_prefix_spec",
    "resolve_minicpm5_route_prefix_reuse",
]
