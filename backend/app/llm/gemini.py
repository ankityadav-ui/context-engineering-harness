from google import genai
from google.genai import types

from .base import BaseLLM


class GeminiLLM(BaseLLM):
    """
    Gemini implementation of the common BaseLLM interface.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
        )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:

        # ----------------------------------------------------
        # Separate system instruction from chat messages
        # ----------------------------------------------------

        system_instruction = None
        contents = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                system_instruction = content

            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=content
                            )
                        ],
                    )
                )

            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                text=content
                            )
                        ],
                    )
                )

        # ----------------------------------------------------
        # Build generation configuration
        # ----------------------------------------------------

        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if temperature is not None:
            config_kwargs["temperature"] = temperature

        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = None

        if config_kwargs:
            config = types.GenerateContentConfig(
                **config_kwargs
            )

        # ----------------------------------------------------
        # Generate response
        # ----------------------------------------------------

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return response.text