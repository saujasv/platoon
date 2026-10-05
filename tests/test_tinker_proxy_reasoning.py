"""Thinking round-trips through the Tinker proxy via ``reasoning_content``.

The proxy hands thinking back to the agent in ``reasoning_content`` and turns it back
into a ThinkingPart when the agent echoes the message, so a multi-turn conversation
re-renders to the tokens that were actually sampled. Uses the real ``qwen3`` renderer
and the Qwen3-8B tokenizer; skipped when either is unavailable.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

import litellm  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402

from platoon.train.tinker.proxy import TinkerLLM, TinkerLLMProxySession  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"
LITELLM_MODEL = f"platoon-tinker/{MODEL_NAME}"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL_NAME} tokenizer is not cached locally")


class ScriptedSamplingClient:
    """Returns each scripted completion in turn, encoded with the real tokenizer."""

    def __init__(self, tokenizer: Any, completions: list[str]) -> None:
        self.tokenizer = tokenizer
        self.completions = list(completions)

    async def sample_async(self, *, prompt: Any, sampling_params: Any, num_samples: int) -> Any:
        tokens = self.tokenizer.encode(self.completions.pop(0), add_special_tokens=False)
        sequence = SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason="stop")
        return SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0)


def _make_llm(
    tokenizer: Any, completions: list[str], *, strip_thinking: bool, renderer_name: str = "qwen3"
) -> TinkerLLM:
    renderer = get_renderer(renderer_name, tokenizer)
    renderer.strip_thinking_from_history = strip_thinking
    return TinkerLLM(
        model_name=MODEL_NAME,
        renderer=renderer,
        tokenizer=tokenizer,
        sampling_client=ScriptedSamplingClient(tokenizer, completions),  # type: ignore[arg-type]
    )


@pytest.fixture
def register_provider():
    """Install a TinkerLLM as LiteLLM's custom provider, restoring the global map after."""
    saved = list(litellm.custom_provider_map)

    def _register(llm: TinkerLLM) -> TinkerLLM:
        return llm.rewrite_litellm_custom_providers()

    yield _register
    litellm.custom_provider_map = saved


async def _run_conversation(
    llm: TinkerLLM, follow_ups: list[dict[str, Any]], **completion_kwargs: Any
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Play one assistant turn per follow-up, echoing replies the way TeamAgent does."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You craft items."},
        {"role": "user", "content": "Craft a wooden plank."},
    ]
    replies: list[dict[str, Any]] = []
    async with TinkerLLMProxySession() as session:
        for follow_up in [*follow_ups, None]:
            response = await litellm.acompletion(
                model=LITELLM_MODEL, messages=messages, api_key="unused", **completion_kwargs
            )
            reply = response.choices[0].message.model_dump(exclude_none=True)
            replies.append(reply)
            if follow_up is None:
                break
            messages = [*messages, reply, follow_up]
        interactions = list(session.interactions.values())
    return interactions, replies


def _assert_each_turn_extends_the_last(interactions: list[Any]) -> None:
    for prev, nxt in zip(interactions, interactions[1:]):
        sampled = prev.obs.to_ints() + list(prev.action.tokens)
        assert nxt.obs.to_ints()[: len(sampled)] == sampled


def test_split_and_rebuild_are_inverse(tokenizer: Any) -> None:
    llm = _make_llm(tokenizer, [], strip_thinking=False)
    parts = [
        {"type": "thinking", "thinking": "\nplan\n"},
        {"type": "text", "text": "\n\nanswer"},
    ]
    content, reasoning = llm._split_reasoning(parts)
    assert (content, reasoning) == ("\n\nanswer", "\nplan\n")
    assert llm._with_thinking_part(reasoning, content) == parts


def test_split_leaves_later_thinking_inline(tokenizer: Any) -> None:
    llm = _make_llm(tokenizer, [], strip_thinking=False)
    assert llm._split_reasoning("plain") == ("plain", None)
    assert llm._split_reasoning([{"type": "text", "text": "a"}, {"type": "thinking", "thinking": "b"}]) == (
        "ab",
        None,
    )
    # An empty think block is still a think block: it has to be re-rendered.
    assert llm._split_reasoning([{"type": "thinking", "thinking": ""}, {"type": "text", "text": "x"}]) == ("x", "")


def test_reasoning_content_reaches_the_agent(tokenizer: Any, register_provider: Any) -> None:
    llm = register_provider(
        _make_llm(tokenizer, ["<think>\nNeed logs first.\n</think>\n\nCrafting now.<|im_end|>"], strip_thinking=False)
    )
    _, replies = asyncio.run(_run_conversation(llm, []))
    assert replies[0]["reasoning_content"] == "\nNeed logs first.\n"
    assert replies[0]["content"] == "\n\nCrafting now."


def test_history_re_renders_to_the_sampled_tokens(tokenizer: Any, register_provider: Any) -> None:
    completions = [
        "<think>\nNeed logs first.\n</think>\n\nI will gather logs.<|im_end|>",
        "<think>\n\n</think>\n\nNow the plank.<|im_end|>",
        "Done.<|im_end|>",
    ]
    llm = register_provider(_make_llm(tokenizer, completions, strip_thinking=False))
    follow_ups = [
        {"role": "user", "content": "Logs gathered."},
        {"role": "user", "content": "Plank crafted."},
    ]
    interactions, _ = asyncio.run(_run_conversation(llm, follow_ups))
    assert len(interactions) == 3
    _assert_each_turn_extends_the_last(interactions)


def _check_tool_call_turn_re_renders(
    tokenizer: Any, register_provider: Any, renderer_name: str, before_call: str = "\n\n"
) -> None:
    call = json.dumps({"name": "craft", "arguments": {"item": "planks"}})
    completions = [
        f"<think>\nUse the craft tool.\n</think>{before_call}<tool_call>\n{call}\n</tool_call><|im_end|>",
        "<think>\nIt worked.\n</think>\n\nCrafted.<|im_end|>",
    ]
    llm = register_provider(_make_llm(tokenizer, completions, strip_thinking=False, renderer_name=renderer_name))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "craft",
                "description": "Craft an item.",
                "parameters": {"type": "object", "properties": {"item": {"type": "string"}}},
            },
        }
    ]
    interactions, replies = asyncio.run(_run_conversation(llm, [], tools=tools))
    assert replies[0]["tool_calls"][0]["function"]["name"] == "craft"
    assert replies[0]["reasoning_content"] == "\nUse the craft tool.\n"
    # Re-run with the tool result echoed back to check the prefix across a tool turn.
    llm.sampling_client = ScriptedSamplingClient(tokenizer, completions)  # type: ignore[assignment]
    tool_result = {"role": "tool", "tool_call_id": replies[0]["tool_calls"][0]["id"], "content": "ok"}
    interactions, _ = asyncio.run(_run_conversation(llm, [tool_result], tools=tools))
    _assert_each_turn_extends_the_last(interactions)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "qwen3 renderer, not thinking: parse_response leaves the '\\n\\n' before <tool_call> in "
        "content, and render_message always adds another '\\n' before the tool calls"
    ),
)
def test_tool_call_turns_re_render_to_the_sampled_tokens(tokenizer: Any, register_provider: Any) -> None:
    _check_tool_call_turn_re_renders(tokenizer, register_provider, "qwen3")


@pytest.mark.parametrize("before_call", ["\n\n", "\n", "", "\n\nCrafting.\n"])
def test_custom_renderer_re_renders_tool_call_turns(
    tokenizer: Any, register_provider: Any, before_call: str
) -> None:
    _check_tool_call_turn_re_renders(tokenizer, register_provider, "qwen3_custom", before_call)


def test_strip_thinking_from_history_still_drops_it(tokenizer: Any, register_provider: Any) -> None:
    completions = [
        "<think>\nSecret plan.\n</think>\n\nOn it.<|im_end|>",
        "Done.<|im_end|>",
    ]
    llm = register_provider(_make_llm(tokenizer, completions, strip_thinking=True))
    interactions, _ = asyncio.run(_run_conversation(llm, [{"role": "user", "content": "Go on."}]))
    assert "Secret plan." not in tokenizer.decode(interactions[1].obs.to_ints())
