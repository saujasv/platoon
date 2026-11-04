"""Renderers registered with ``tinker_cookbook`` under platoon-specific names.

Importing this module registers them, so any ``get_renderer`` call made after
``platoon.train.tinker.proxy`` is imported can resolve these names.
"""

from __future__ import annotations

import json

import tinker
from tinker_cookbook.renderers import Qwen3Renderer, register_renderer
from tinker_cookbook.renderers.base import (
    ImagePart,
    Message,
    RenderContext,
    RenderedMessage,
    TextPart,
    _tool_call_payload,
    remove_thinking,
)
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

QWEN3_CUSTOM = "qwen3_custom"
QWEN3_5_CUSTOM = "qwen3_5_custom"


class Qwen3CustomRenderer(Qwen3Renderer):
    """``qwen3`` without the newline it inserts before an assistant turn's tool calls.

    ``parse_response`` keeps the whitespace the model wrote before ``<tool_call>`` in
    the message content, and ``Qwen3Renderer.render_message`` then adds its own
    ``"\\n"`` on top to match the HF template, so a sampled
    ``</think>\\n\\n<tool_call>`` re-renders as ``</think>\\n\\n\\n<tool_call>`` and
    the next prompt no longer extends the sampled tokens. Here the content is
    trusted to carry that whitespace already.

    ``render_message`` is otherwise a copy of ``Qwen3Renderer.render_message``.
    """

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        content = message["content"]

        if isinstance(content, list):
            parts = content
            if self.strip_thinking_from_history and message["role"] == "assistant" and not ctx.is_last:
                parts = remove_thinking(parts)
            rendered_parts = []
            for p in parts:
                if p["type"] == "thinking":
                    rendered_parts.append(f"<think>{p['thinking']}</think>")
                elif p["type"] == "text":
                    rendered_parts.append(p["text"])
            output_content = "".join(rendered_parts)
        else:
            output_content = content

        if message["role"] == "tool":
            output_content = self._wrap_qwen_tool_response(output_content)

        if "tool_calls" in message:
            # The only change from Qwen3Renderer: no "\n" before the first call.
            output_content += "\n".join(
                f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                for tool_call in message["tool_calls"]
            )
        output_content += "<|im_end|>"
        header = tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(header_str, add_special_tokens=False))
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_content, add_special_tokens=False))
        ]
        return RenderedMessage(header=header, output=output)


register_renderer(QWEN3_CUSTOM, lambda tokenizer, image_processor=None: Qwen3CustomRenderer(tokenizer))


class Qwen3_5CustomRenderer(Qwen3_5Renderer):
    """``qwen3_5`` that re-renders sampled tool-call turns to the tokens that were sampled.

    Thinking is kept in history by default, since dropping it rewrites every earlier
    turn and no prompt can extend the last. With it kept, ``qwen3_5`` still breaks the
    extension on tool-call turns in two ways, both whitespace:

    * A sampled ``</think>\\n\\n<tool_call>`` re-renders as ``</think>\\n\\n\\n\\n<tool_call>``:
      the thinking block already ends in ``"\\n\\n"`` and ``_format_tool_calls_chunks``
      prepends another. It is dropped when the turn has thinking and no text.
    * The ``"\\n"`` between parallel calls is kept by ``parse_response`` as whitespace-only
      text, which renders before the first call while the formatter puts the separators
      back between them. Such text is dropped from tool-call turns.

    A model that departs from its own format, e.g. no newline before ``</parameter>``,
    still re-renders differently; that needs the raw sampled text, not a renderer.
    """

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        if message["role"] == "assistant" and message.get("tool_calls"):
            content = message["content"]
            if isinstance(content, list):
                content = [p for p in content if not (p["type"] == "text" and not p["text"].strip())]
            elif not content.strip():
                content = ""
            message = {**message, "content": content}
        return super().render_message(message, ctx)

    def _format_tool_calls_chunks(self, message: Message) -> list[ImagePart | TextPart]:
        chunks = super()._format_tool_calls_chunks(message)
        content = message["content"]
        thinking_only = isinstance(content, list) and all(p["type"] == "thinking" for p in content) and bool(content)
        if thinking_only and chunks[0]["text"].startswith("\n\n"):
            chunks[0] = TextPart(type="text", text=chunks[0]["text"][2:])
        return chunks


register_renderer(
    QWEN3_5_CUSTOM,
    lambda tokenizer, image_processor=None: Qwen3_5CustomRenderer(
        tokenizer, image_processor=image_processor, strip_thinking_from_history=False
    ),
)
