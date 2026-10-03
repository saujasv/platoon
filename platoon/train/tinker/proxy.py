# Adapted and modified from: https://github.com/microsoft/agent-lightning/blob/main/examples/tinker/agl_tinker/llm.py

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Type,
    TypeGuard,
    TypeVar,
    cast,
    get_origin,
)

import litellm
import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
    Choices,
    ModelResponse,
    Usage,
)
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import TopLogprob as LitellmTopLogprob
from litellm.utils import custom_llm_setup
from tinker.types import ModelInput, SampleResponse, SamplingParams
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from tinker_cookbook.renderers import ToolSpec as TinkerToolSpec
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoProcessor, PreTrainedTokenizer

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TinkerLLMInteraction:
    obs: tinker.ModelInput
    action: TokensWithLogprobs
    # Prompt tokens Tinker billed as prefix-cache hits, straight off the
    # ``SampleResponse``. ``obs`` is the whole conversation resent every turn, so
    # ``obs.length`` alone overstates what was charged; this is the discount.
    # Defaulted so interactions built without it stay valid.
    prompt_cache_hit_tokens: int = 0


proxy_interactions: ContextVar[dict[str, TinkerLLMInteraction]] = ContextVar("proxy_interactions")


class TinkerLLMProxySession:
    _token: object | None = None

    def __enter__(self) -> TinkerLLMProxySession:
        self._token = proxy_interactions.set({})
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is not None:
            proxy_interactions.reset(self._token)
            self._token = None

    async def __aenter__(self) -> TinkerLLMProxySession:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return self.__exit__(exc_type, exc_value, traceback)

    @property
    def interactions(self) -> dict[str, TinkerLLMInteraction]:
        return proxy_interactions.get()


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: String prefix for the generated ID.

    Returns:
        A unique identifier string.
    """
    return prefix + str(uuid.uuid4())


class TinkerLLM(CustomLLM):
    """LiteLLM provider that proxies Tinker's sampling client.

    Attributes:
        model_name: The HuggingFace model identifier.
        renderer: Prompt renderer for formatting messages.
        tokenizer: Tokenizer for the model.
        sampling_client: Tinker sampling client for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        model_name: str,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizer,
        sampling_client: tinker.SamplingClient,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        seed: int = 42,
        context_window_length: int | None = None,
    ) -> None:
        """Initialize the TinkerLLM."""
        self.model_name = model_name
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.context_window_length = context_window_length
        self._version: int = 0

    @property
    def version(self) -> int:
        """Get the current checkpoint version."""
        return self._version

    def set_version(self, version: int) -> None:
        """Set the checkpoint version.

        Args:
            version: The version number to set.
        """
        self._version = version

    def increment_version(self) -> int:
        """Increment and return the checkpoint version.

        Returns:
            The new version number after incrementing.
        """
        self._version += 1
        return self._version

    def update_sampling_client(self, sampling_client: tinker.SamplingClient, increment_version: bool = True) -> None:
        """Update the sampling client used for generation.

        Args:
            sampling_client: New Tinker sampling client to use.
            increment_version: Whether to increment the version after update.
        """
        self.sampling_client = sampling_client
        if increment_version:
            self.increment_version()

    @staticmethod
    def _canonicalize_tool_call(raw: Any) -> TinkerToolCall:
        """One LiteLLM/OpenAI tool call -> the renderer's ``ToolCall`` model.

        LiteLLM carries tool calls as OpenAI wire dicts, but renderers reach into
        them by attribute (``tc.function.name``), so a dict reaches the renderer as
        an ``AttributeError`` rather than a useful message. ``ToolCall`` is
        ``extra="forbid"``, so this builds the model field by field instead of
        validating the dict: LiteLLM adds keys such as ``index`` that would
        otherwise be rejected.
        """
        if isinstance(raw, TinkerToolCall):
            return raw
        function = raw.get("function") if isinstance(raw, dict) else getattr(raw, "function", None)
        if function is None:
            raise ValueError(f"Tool call has no function body: {raw!r}")
        if isinstance(function, dict):
            name, arguments = function.get("name"), function.get("arguments")
        else:
            name, arguments = getattr(function, "name", None), getattr(function, "arguments", None)
        if not name:
            raise ValueError(f"Tool call has no function name: {raw!r}")
        call_id = raw.get("id") if isinstance(raw, dict) else getattr(raw, "id", None)
        return TinkerToolCall(
            id=call_id,
            # The renderers emit this verbatim into the prompt and tools parse it as
            # JSON, so an absent argument list has to be an empty object.
            function=TinkerToolCall.FunctionBody(name=str(name), arguments=str(arguments or "{}")),
        )

    def _canonicalize_messages(self, messages: Any) -> List[TinkerMessage]:
        # Note: We avoid using TypeAdapter for strict validation because TinkerMessage
        # contains ImagePart which has PIL.Image.Image that Pydantic can't handle.
        # Instead, we cast directly since we expect the messages to already be in
        # the correct format (coming from LiteLLM or manually constructed).
        if not isinstance(messages, list):
            raise ValueError(f"Expected list of messages, got {type(messages)}")
        # Two fields the cast cannot cover, both on assistant turns that call tools:
        #
        # ``tool_calls`` -- renderers read these by attribute, so LiteLLM's OpenAI
        #   wire dicts have to become real models first.
        # ``content`` -- renderers index it unconditionally, but LiteLLM builds the
        #   message with ``model_dump(exclude_none=True)``, so a turn that is only a
        #   tool call arrives with no ``content`` key at all.
        #
        # Rebuild only the messages that need it, leaving every other one identical.
        canonical: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                canonical.append(message)
                continue
            tool_calls = message.get("tool_calls")
            needs_content = message.get("role") == "assistant" and "content" not in message
            if not tool_calls and not needs_content:
                canonical.append(message)
                continue
            rebuilt = dict(message)
            if tool_calls:
                rebuilt["tool_calls"] = [self._canonicalize_tool_call(tc) for tc in tool_calls]
            if needs_content:
                rebuilt["content"] = ""
            canonical.append(rebuilt)
        return cast(List[TinkerMessage], canonical)

    def _validate_role(self, role: str) -> TypeGuard[Literal["assistant", "user", "system", "tool", "function"]]:
        if role not in ["assistant", "user", "system", "tool", "function"]:
            raise ValueError(f"Invalid role: {role}")
        return True

    def _parse_tool_call(self, tool_call: TinkerToolCall) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=tool_call.id or generate_id("tinker-tool-call-"),
            function={
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
            type="function",
        )

    def _normalize_message_content(self, content: Any) -> str:
        """Convert tinker-cookbook structured content parts to plain text for LiteLLM."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        parts.append(str(part.get("text", "")))
                    elif part_type == "thinking":
                        parts.append(str(part.get("thinking", "")))
                    elif part_type == "image":
                        # LiteLLM's Message model expects plain string content on this path.
                        # Drop image payloads but keep a marker so content is not silently empty.
                        parts.append("[image]")
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)

    def _get_optional_params(
        self,
        kwargs: Dict[str, Any],
        keys: List[str],
        expected_type: Type[T],
        validate_fn: Callable[[T], bool],
        default_value: T,
    ) -> T:
        optional_params = cast(Dict[str, Any], kwargs.get("optional_params", {}))
        if not isinstance(optional_params, dict):  # type: ignore
            raise ValueError(f"Invalid optional params type: {type(optional_params)}")
        for key in keys:
            if key in optional_params:
                value = optional_params[key]
                # Handle parameterized generics like list[str] by extracting the origin type
                origin = get_origin(expected_type)
                check_type = origin if origin is not None else expected_type
                # Allow int for float params (e.g., top_p=1 instead of top_p=1.0)
                if check_type is float:
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Invalid {key} type: {type(value)}")
                    value = float(value)  # Convert int to float
                elif not isinstance(value, check_type):
                    raise ValueError(f"Invalid {key} type: {type(value)}")
                if not validate_fn(value):
                    raise ValueError(f"Invalid {key}. Did not pass validation: {value}")
                return value
        return default_value

    @staticmethod
    def _canonicalize_tool_specs(tools: Any) -> list[TinkerToolSpec]:
        """LiteLLM's ``tools`` -> the renderer's flat ``ToolSpec`` list.

        LiteLLM passes OpenAI's nested envelope, ``{"type": "function", "function":
        {...}}``, while ``ToolSpec`` is flat, so the body has to be lifted out. A
        spec that is already flat is taken as-is.
        """
        specs: list[TinkerToolSpec] = []
        for tool in tools or []:
            body = tool.get("function", tool) if isinstance(tool, dict) else None
            if not isinstance(body, dict) or not body.get("name"):
                raise ValueError(f"Tool spec has no function name: {tool!r}")
            specs.append(
                cast(
                    TinkerToolSpec,
                    {
                        "name": body["name"],
                        "description": body.get("description", ""),
                        "parameters": body.get("parameters", {}),
                    },
                )
            )
        return specs

    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
        canonical_messages = self._canonicalize_messages(messages)
        # LiteLLM routes request options into ``optional_params`` rather than
        # leaving them top level, which is why the sampling knobs below are read
        # through ``_get_optional_params``. ``tools`` arrives the same way; the
        # top-level lookup only covers a handler invoked directly.
        optional_params = kwargs.get("optional_params")
        tools = kwargs.get("tools")
        if tools is None and isinstance(optional_params, dict):
            tools = optional_params.get("tools")
        specs = self._canonicalize_tool_specs(tools)
        if specs:
            canonical_messages = self._prepend_tool_prefix(canonical_messages, specs)
        return self.renderer.build_generation_prompt(canonical_messages)

    def _prepend_tool_prefix(
        self, messages: List[TinkerMessage], specs: list[TinkerToolSpec]
    ) -> List[TinkerMessage]:
        """Put the renderer's tool-definition prefix at the front of a conversation.

        ``build_generation_prompt`` takes no ``tools`` argument: a renderer declares
        tools by way of ``create_conversation_prefix_with_tools``, which folds the
        system prompt and the tool schemas into whatever messages that format needs
        (for Harmony, a routing system message plus a developer message carrying the
        ``functions`` namespace). Without this the schemas never reach the model and
        it has to guess every signature.

        The prefix *absorbs* the system prompt rather than sitting beside it, so the
        leading system message is handed over and dropped from the tail.
        """
        head, tail = messages, []
        system_prompt = ""
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system_prompt = self._normalize_message_content(messages[0].get("content"))
            tail = list(messages[1:])
        else:
            tail = list(messages)
        del head
        try:
            prefix = self.renderer.create_conversation_prefix_with_tools(specs, system_prompt)
        except NotImplementedError:
            # This renderer has no tool format. Leave the conversation untouched so
            # behaviour matches a plain text model rather than failing the request.
            logger.warning(
                "Renderer %s does not support tool definitions; %s tool(s) omitted from the prompt",
                type(self.renderer).__name__,
                len(specs),
            )
            return messages
        return cast(List[TinkerMessage], list(prefix) + tail)

    def _parse_response(self, model_input: ModelInput, response: SampleResponse) -> ModelResponse:
        """Tinker Response -> LiteLLM Response.

        Extract log probabilities as well.
        """
        choices: List[Choices] = []
        completion_token_count = 0
        for seq in response.sequences:
            completion_token_count += len(seq.tokens)
            if seq.logprobs is not None:
                token_strings: List[str] = self.tokenizer.batch_decode([[token] for token in seq.tokens])  # type: ignore
                # FIXME: This might not be accurate for some corner cases.
                # But it's not actually used in most cases.
                bytes_list: List[List[int]] = [list(token.encode("utf-8")) for token in token_strings]
                logprobs = LitellmChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=token,
                            bytes=bytes,
                            logprob=logprob,
                            # NOTE: This top logprob is fake. It's just used to fool the LiteLLM type validator.
                            top_logprobs=[LitellmTopLogprob(token=token, bytes=bytes, logprob=logprob)],
                        )
                        for token, bytes, logprob in zip(token_strings, bytes_list, seq.logprobs)
                    ]
                )
            else:
                logprobs = None

            parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    assert False, "This should never happen"
                # FIXME: The content should not be still there if tool call has been parsed.
                content = self._normalize_message_content(parsed_response["content"])
                # NOTE(yuge): I thought about adding this to make it more robust to empty responses,
                # but later I found it's a configuration error in my renderer. So I think it's better
                # to just log a warning and go with the default path.
                # if not content:
                #     raise ValueError("Parsed content is empty. Original response: " + str(response))
                if not content:
                    logger.warning("Parsed content is empty. Original response: " + str(response))
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]
                choices.append(
                    Choices(
                        message=LitellmMessage(role=role, content=content, tool_calls=tool_calls),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
            else:
                # logger.warning(f"Failed to parse response: {parsed_response}")
                # Go with the default path
                choices.append(
                    Choices(
                        message=LitellmMessage(
                            role="assistant",
                            content=self._normalize_message_content(parsed_response["content"]),
                        ),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
        prompt_token_count = model_input.length
        return ModelResponse(
            id=generate_id("tinker-sampling-"),
            model=self.model_name,
            choices=choices,
            prompt_token_ids=model_input.to_ints(),
            usage=Usage(
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_token_count,
                total_tokens=prompt_token_count + completion_token_count,
            ),
        )

    def _record_interaction(
        self,
        model_input: ModelInput,
        model_response: ModelResponse,
        response: SampleResponse | None = None,
    ) -> None:
        assert len(model_response.choices) == 1

        logprobs_content = model_response.choices[0].logprobs.content
        interaction = TinkerLLMInteraction(
            obs=model_input,
            action=TokensWithLogprobs(
                tokens=model_response.choices[0].token_ids,
                maybe_logprobs=[c.logprob for c in logprobs_content] if logprobs_content else [],
            ),
            # Taken from the raw response rather than routed through the LiteLLM
            # ``Usage``, whose other fields this proxy computes itself anyway.
            prompt_cache_hit_tokens=int(getattr(response, "prompt_cache_hit_tokens", 0) or 0),
        )
        proxy_interactions.get()[model_response.id] = interaction

    def _check_context_window_length(self, model_input: ModelInput, max_completion_tokens: int) -> None:
        prompt_length = model_input.length
        total_sequence_length = prompt_length + max_completion_tokens
        if self.context_window_length is not None and total_sequence_length > self.context_window_length:
            raise ValueError(
                f"Prompt length plus max_tokens exceeds the model's context window: "
                f"{prompt_length} prompt tokens + {max_completion_tokens} max_tokens > "
                f"{self.context_window_length} context window length."
            )

    async def acompletion(self, **kwargs: Any) -> ModelResponse:  # type: ignore
        """Main entrypoint for LiteLLM to call."""
        import asyncio
        import time

        max_tokens = self._get_optional_params(
            kwargs, ["max_completion_tokens", "max_tokens"], int, lambda x: x >= 0, self.max_tokens
        )
        temperature = self._get_optional_params(
            kwargs, ["temperature"], float, lambda x: 0.0 <= x <= 2.0, self.temperature
        )
        top_k = self._get_optional_params(kwargs, ["top_k"], int, lambda x: True, self.top_k)
        top_p = self._get_optional_params(kwargs, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p)
        seed = self._get_optional_params(kwargs, ["seed"], int, lambda _: True, self.seed)
        stop_sequences = self._get_optional_params(
            kwargs, ["stop"], list, lambda x: True, self.renderer.get_stop_sequences()
        )
        model_input = self._prepare_model_input(**kwargs)
        self._check_context_window_length(model_input, max_tokens)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences,
        )
        start_time = time.perf_counter()

        # Timeout for sample_async to prevent infinite hangs (10 minutes)
        SAMPLE_TIMEOUT_SECONDS = 600
        try:
            result = await asyncio.wait_for(
                self.sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1),
                timeout=SAMPLE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.error(f"sample_async timed out after {elapsed:.1f}s (timeout={SAMPLE_TIMEOUT_SECONDS}s)")
            raise TimeoutError(f"Tinker sample_async timed out after {SAMPLE_TIMEOUT_SECONDS}s")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.exception(f"sample_async failed after {elapsed:.1f}s: {e}")
            raise
        elapsed = time.perf_counter() - start_time
        if elapsed > 30.0:
            logger.warning(f"sample_async took {elapsed:.1f}s (slow)")
        final_response = self._parse_response(model_input, result)
        self._record_interaction(model_input, final_response, result)
        return final_response

    def completion(self, **kwargs: Any) -> ModelResponse:  # type: ignore
        """Main entrypoint for LiteLLM to call."""
        max_tokens = self._get_optional_params(
            kwargs, ["max_completion_tokens", "max_tokens"], int, lambda x: x >= 0, self.max_tokens
        )
        temperature = self._get_optional_params(
            kwargs, ["temperature"], float, lambda x: 0.0 <= x <= 2.0, self.temperature
        )
        top_k = self._get_optional_params(kwargs, ["top_k"], int, lambda x: True, self.top_k)
        top_p = self._get_optional_params(kwargs, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p)
        seed = self._get_optional_params(kwargs, ["seed"], int, lambda _: True, self.seed)
        stop_sequences = self._get_optional_params(
            kwargs, ["stop"], list, lambda x: True, self.renderer.get_stop_sequences()
        )
        model_input = self._prepare_model_input(**kwargs)
        self._check_context_window_length(model_input, max_tokens)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences,
        )
        result = self.sampling_client.sample(prompt=model_input, sampling_params=params, num_samples=1)
        final_response = self._parse_response(model_input, result)
        self._record_interaction(model_input, final_response, result)
        return final_response

    def as_model_list(self) -> list[dict]:
        """Generate model configuration for LiteLLM proxy.

        Returns:
            List containing model configuration dict for LiteLLM.
        """
        return [
            {
                "model_name": self.model_name,
                "litellm_params": {
                    "model": f"platoon-tinker/{self.model_name}",
                },
            }
        ]

    def rewrite_litellm_custom_providers(self) -> TinkerLLM:
        """Register this TinkerLLM as a custom provider in LiteLLM.

        !!! warning
            This method modifies the global LiteLLM state, which could interfere with other tests in the
            same process.

        Returns:
            Self for method chaining.
        """
        litellm.custom_provider_map = [{"provider": "platoon-tinker", "custom_handler": self}]
        custom_llm_setup()
        return self


@dataclass
class ModelInfo:
    llm: TinkerLLM
    model_name: str
    base_url: str
    api_key: str


def register_tinker_llm(
    model_name: str,
    renderer_name: str,
    context_window_length: int | None = None,
    renderer_kwargs: dict[str, Any] | None = None,
) -> ModelInfo:
    """
    Register the TinkerLLMProxy as a custom provider in LiteLLM.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507").
        renderer_name: Renderer type for prompt formatting (e.g., "qwen3", "qwen3_instruct").
        context_window_length: Context window length for the model. Defaults to None.
        renderer_kwargs: Optional renderer attribute overrides applied after construction.
    """
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    image_processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.debug("Could not load AutoProcessor for %s: %s", model_name, e)
    else:
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None:
            tokenizer = processor_tokenizer
        image_processor = getattr(processor, "image_processor", None)

    renderer = get_renderer(renderer_name, tokenizer, image_processor=image_processor, model_name=model_name)
    for key, value in (renderer_kwargs or {}).items():
        if not hasattr(renderer, key):
            raise ValueError(f"Renderer '{renderer_name}' does not support renderer_kwargs['{key}']")
        setattr(renderer, key, value)

    tinker_llm = TinkerLLM(
        model_name=model_name,
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        context_window_length=context_window_length,
    )
    tinker_llm.rewrite_litellm_custom_providers()
    base_url = "None"
    api_key = "None"
    model_name = "platoon-tinker/" + model_name
    return ModelInfo(llm=tinker_llm, model_name=model_name, base_url=base_url, api_key=api_key)
