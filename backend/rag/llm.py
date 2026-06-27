import asyncio
import logging

from groq import Groq

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Set it in .env or the environment."
            )
        self.client = Groq(api_key=settings.groq_api_key)
        logger.info("Using Groq LLM model %s", settings.llm_model)

    def _generate(self, messages: list[dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response")
        return content.strip()

    async def generate_answer(self, messages: list[dict[str, str]]) -> str:
        try:
            return await asyncio.to_thread(self._generate, messages)
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise RuntimeError(f"LLM generation failed: {exc}") from exc
