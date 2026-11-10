"""An EMPTY native generation is an empty response; malformed output is not."""
from __future__ import annotations

import unittest
from pathlib import Path

from orbit.backend.base import ChatResult
from orbit.native_llama.events import NativeTimings
from orbit.runtime.chat import ChatRuntime
from tests import test_native_qwen_profile as qwen_tests
from unittest import mock


def _result(content: str) -> ChatResult:
    return ChatResult(content=content, model="fake", finish_reason="stop", tool_calls=[],
                      prompt_tokens=3, completion_tokens=0, cached_tokens=0,
                      prompt_tokens_per_second=None, generation_tokens_per_second=None)


class NativeClientEmptyGenerationTests(unittest.TestCase):
    def _complete(self, raw: str, *, output_tokens: int = 0, max_tokens: int = 64):
        client = qwen_tests.NativeQwenProfileTests()._client()

        def generate(_messages, **kwargs):
            if raw:
                kwargs["on_token"](raw)
            return NativeTimings(10, output_tokens, 0, 10, 1, 1)

        def strict_parse(text, *, partial):
            raise RuntimeError("The model produced output that does not match the expected peg-native format")

        with mock.patch.object(client, "complete_chat", side_effect=generate), \
             mock.patch.object(client, "_parse_profile_output", side_effect=strict_parse):
            return client._complete_profile_chat_text_once(
                [{"role": "user", "content": "x"}], max_tokens=max_tokens, stop=(),
                tools=[{"type": "function", "function": {"name": "x"}}], thinking=False,
                route_prefix_anchor=False, qwen_route_prefix_anchor=False,
                qwen36_shell_tool_prefix_anchor=False, allow_mtp_experimental=None,
                final_prefix_experiment=False)

    def test_empty_generation_becomes_empty_completion(self):
        completion = self._complete("")
        self.assertEqual((completion.content, completion.reasoning_content, tuple(completion.tool_calls)), ("", "", ()))

    def test_non_empty_malformed_output_still_fails(self):
        for raw in ("<tool_call><function=", "garbage", " "):
            with self.subTest(raw=raw), self.assertRaisesRegex(RuntimeError, "peg-native"):
                self._complete(raw)


class RuntimeEmptyResponseBoundTests(unittest.TestCase):
    def test_persistent_empty_tool_turn_is_bounded(self):
        class Backend:
            def __init__(self): self.calls = 0
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                return _result("")
            def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
                self.calls += 1
                return _result("")
        backend = Backend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.ask_auto("what is the current directory?", temperature=0, max_tokens=64, workdir=Path("."))
        self.assertLessEqual(backend.calls, 6)
        self.assertGreaterEqual(backend.calls, 2)


if __name__ == "__main__":
    unittest.main()
