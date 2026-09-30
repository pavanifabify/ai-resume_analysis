"""
llm_client.py
-------------
A thin, provider-agnostic wrapper around OpenAI and Gemini chat APIs.

The rest of the application does not need to know which LLM provider
is being used.
"""

import json
import re
import time
from typing import Optional

from src.config import Config


class LLMError(Exception):
    """Raised when the LLM call fails after retries."""
    pass


class LLMClient:

    def __init__(self):
        self.provider = Config.LLM_PROVIDER.lower()

        # ==========================================================
        # OPENAI
        # ==========================================================

        if self.provider == "openai":

            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required. "
                    "Install it with: pip install openai"
                ) from e

            if not Config.OPENAI_API_KEY:
                raise LLMError(
                    "OPENAI_API_KEY is missing. "
                    "Add it to your .env file."
                )

            self.client = OpenAI(
                api_key=Config.OPENAI_API_KEY
            )

            self.model = Config.OPENAI_MODEL

        # ==========================================================
        # GEMINI
        # ==========================================================

        elif self.provider == "gemini":

            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError(
                    "The 'google-generativeai' package is required. "
                    "Install it with: pip install google-generativeai"
                ) from e

            if not Config.GEMINI_API_KEY:
                raise LLMError(
                    "GEMINI_API_KEY is missing. "
                    "Add it to your .env file."
                )

            genai.configure(
                api_key=Config.GEMINI_API_KEY
            )

            self.genai = genai
            self.model = Config.GEMINI_MODEL

        # ==========================================================
        # INVALID PROVIDER
        # ==========================================================

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {self.provider}"
            )

    # ==============================================================
    # CHAT
    # ==============================================================

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        retries: int = 2,
    ) -> str:
        """
        Send a normal chat request to the configured provider.

        Returns:
            str: Raw text response from the LLM.
        """

        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):

            try:

                # ==================================================
                # OPENAI
                # ==================================================

                if self.provider == "openai":

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    content = response.choices[0].message.content

                    if not content:
                        raise LLMError(
                            "OpenAI returned an empty response."
                        )

                    return content.strip()

                # ==================================================
                # GEMINI
                # ==================================================

                elif self.provider == "gemini":

                    model = self.genai.GenerativeModel(
                        self.model
                    )

                    prompt = (
                        system_prompt
                        + "\n\n"
                        + user_prompt
                    )

                    response = model.generate_content(
                        prompt
                    )

                    content = response.text

                    if not content:
                        raise LLMError(
                            "Gemini returned an empty response."
                        )

                    return content.strip()

            except Exception as e:

                last_error = e

                if attempt < retries:
                    time.sleep(1.0)

        raise LLMError(
            f"LLM request failed after "
            f"{retries + 1} attempts: {last_error}"
        )

    # ==============================================================
    # CHAT JSON
    # ==============================================================

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        retries: int = 2,
    ) -> dict:
        """
        Send a chat request and parse the response as JSON.

        Handles:
        - Markdown code fences
        - Extra text around JSON
        - JSON parsing errors
        - LLM retries
        """

        json_system_prompt = (
            system_prompt
            + "\n\nCRITICAL INSTRUCTION:"
            + "\nReturn ONLY a valid JSON object."
            + "\nUse double quotes for ALL keys and string values."
            + "\nDo NOT use single quotes."
            + "\nDo NOT include markdown."
            + "\nDo NOT include ```json or ```."
            + "\nDo NOT include explanations before or after the JSON."
        )

        last_err: Optional[Exception] = None

        for attempt in range(retries + 1):

            try:

                # --------------------------------------------------
                # Get response from LLM
                # --------------------------------------------------

                raw = self.chat(
                    json_system_prompt,
                    user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=0,
                )

                cleaned = self._strip_code_fences(
                    raw
                ).strip()

                # --------------------------------------------------
                # First attempt: response itself is JSON
                # --------------------------------------------------

                try:

                    result = json.loads(cleaned)

                    if not isinstance(result, dict):
                        raise ValueError(
                            "LLM returned valid JSON, "
                            "but it was not a JSON object."
                        )

                    return result

                except json.JSONDecodeError as e:

                    last_err = e

                # --------------------------------------------------
                # Second attempt: extract JSON object
                # --------------------------------------------------

                start = cleaned.find("{")
                end = cleaned.rfind("}")

                if (
                    start != -1
                    and end != -1
                    and end > start
                ):

                    json_text = cleaned[
                        start:end + 1
                    ]

                    try:

                        result = json.loads(
                            json_text
                        )

                        if not isinstance(result, dict):
                            raise ValueError(
                                "Extracted JSON is not an object."
                            )

                        return result

                    except (
                        json.JSONDecodeError,
                        ValueError,
                    ) as e:

                        last_err = e

                # --------------------------------------------------
                # Retry
                # --------------------------------------------------

                if attempt < retries:
                    time.sleep(1.0)

            except Exception as e:

                last_err = e

                if attempt < retries:
                    time.sleep(1.0)

        raise ValueError(
            f"Failed to parse JSON from LLM response: "
            f"{last_err}"
        )

    # ==============================================================
    # REMOVE MARKDOWN CODE FENCES
    # ==============================================================

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        Remove markdown code fences such as:

        ```json
        {
            "score": 85
        }
        ```

        Returns only the JSON text.
        """

        text = text.strip()

        if text.startswith("```"):

            # Remove opening ``` or ```json
            text = re.sub(
                r"^```[a-zA-Z0-9_-]*\s*",
                "",
                text,
                count=1,
            )

            # Remove closing ```
            text = re.sub(
                r"\s*```$",
                "",
                text,
                count=1,
            )

        return text.strip()