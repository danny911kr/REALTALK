from enum import StrEnum
import json
from typing import Literal

from loguru import logger
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel
import tiktoken


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class StopReasonEnum(StrEnum):
    LENGTH = "length"
    STOP = "stop"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"  # fallback for future values


class LLMResponse(BaseModel):
    content: str
    model: str
    stop_reason: StopReasonEnum
    token_usage: TokenUsage


class LLMJSONResponse(BaseModel):
    content: dict
    model: str
    stop_reason: StopReasonEnum
    token_usage: TokenUsage


class OpenAIGenerator:
    def __init__(self, oai_api_key: str, model: str):
        self.model = model
        self.sync_client = OpenAI(api_key=oai_api_key)
        self.timeout = 100.0
        self.max_retries = 3

    def _convert_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        message_params: list[ChatCompletionMessageParam] = []
        for message in messages:
            if message.role == "user":
                message_params.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=message.content,
                    )
                )
            elif message.role == "assistant":
                message_params.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=message.content,
                    )
                )
            elif message.role == "system":
                message_params.append(
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content=message.content,
                    )
                )
            else:
                raise ValueError(f"Invalid message role: {message.role}")
        return message_params

    def _convert_stop_reason(
        self,
        stop_reason: str | None,
    ) -> StopReasonEnum:
        if stop_reason is None:
            raise ValueError("stop_reason should not be None.")
        try:
            return StopReasonEnum(stop_reason)
        except ValueError:
            logger.warning(f"Unknown stop reason: {stop_reason}")
            return StopReasonEnum.OTHER

    def _count_tokens(self, messages: list[LLMMessage]):
        """Return the number of tokens used by a list of messages.

        Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        tokens_per_message = 3
        # tokens_per_name = 1
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message.content))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _convert_completion_to_response(
        self,
        completion: ChatCompletion,
    ) -> LLMResponse:
        content = completion.choices[0].message.content or ""
        assert completion.usage is not None, "message.usage should not be None."
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        return LLMResponse(
            content=content,
            model=self.model,
            stop_reason=self._convert_stop_reason(completion.choices[0].finish_reason),
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    def generate(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        message_params = self._convert_messages(messages)
        try:
            subscription = self.sync_client.with_options(
                timeout=self.timeout,
                max_retries=self.max_retries,
            ).chat.completions.create(
                model=self.model,
                messages=message_params,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"{e}")
            raise e

        assert isinstance(subscription, ChatCompletion), "Response should be returned in bulk."
        return self._convert_completion_to_response(
            completion=subscription,
        )

    def generate_json(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMJSONResponse:
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = self.generate(
                    messages=messages,
                    response_format={"type": "json_object"},
                    **kwargs,
                )
                response_json = json.loads(response.content)
                return LLMJSONResponse(
                    content=response_json,
                    model=response.model,
                    stop_reason=response.stop_reason,
                    token_usage=response.token_usage,
                )
            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(f"[generate_json] JSON decoding failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    logger.info("[generate_json] Retrying ...")
                else:
                    logger.error("[generate_json] Max retries reached.")
            except Exception as e:
                logger.error(f"[generate_json] Unexpected error: {e}")
                raise e

        raise last_exception if last_exception else RuntimeError("Unknown error during JSON decoding")
