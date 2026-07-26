"""
=========================================================
Gemini Service

Purpose
-------
Send a prepared list of prompt parts (text + cropped defect
images) to Gemini Vision and parse the JSON explanation array
it returns.

Reliability notes (read before changing retry logic)
------------------------------------------------------
Two independent layers of retry are in play here, for two
different failure modes:

1. Transport-level (503 busy, 429 rate-limited, timeouts) —
   handled by the google-genai SDK's own HttpRetryOptions. This
   is more correct than a hand-rolled sleep loop because the
   SDK knows which errors are safe to retry and backs off
   per-request, not per our own guesswork.

2. Logical-level (200 OK, but the model returned malformed
   JSON, an empty response, or was blocked by a safety filter)
   — the SDK can't retry these because from its point of view
   the request succeeded. We retry these ourselves, since LLM
   JSON-formatting hiccups are usually transient and a second
   attempt commonly succeeds.

Model selection
----------------
The model id is configurable via GEMINI_MODEL so a model
rename/deprecation on Google's side is a one-line .env change,
not a code change. Defaults to gemini-3.6-flash — Google's
current GA Flash-tier model with the strongest multimodal/
spatial reasoning in that tier as of mid-2026, at a lower
output-token price than the previous gemini-3.5-flash. Pin an
older model in .env if you hit any regressions.

Author : Jasmeen
=========================================================
"""

from __future__ import annotations

import json
import os
import random
import time

import cv2
import numpy as np
from PIL import Image

from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"

# Retried automatically at the transport layer by the SDK.
RETRYABLE_HTTP_STATUS = [429, 500, 502, 503, 504]


class GeminiService:

    def __init__(self, model: str | None = None, max_attempts: int = 3, timeout_sec: float = 60.0):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("Gemini API Key not found.")

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.max_attempts = max_attempts

        http_options = types.HttpOptions(
            timeout=int(timeout_sec * 1000),
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=15.0,
                exp_base=2.0,
                http_status_codes=RETRYABLE_HTTP_STATUS,
            ),
        )

        try:
            self.client = genai.Client(api_key=api_key, http_options=http_options)
        except TypeError:
            # Defensive fallback for a google-genai version whose
            # HttpOptions/HttpRetryOptions shape differs from the one
            # this was written against — better to run without
            # SDK-level retry/timeout tuning than to fail to start.
            print("Gemini SDK does not support the configured http_options; using client defaults.")
            self.client = genai.Client(api_key=api_key)

        print(f"Using Gemini Model : {self.model}")

    # =====================================================
    # Content assembly
    # =====================================================

    @staticmethod
    def _to_parts(contents: list) -> list:
        """Converts a mixed list of str / BGR numpy image arrays
        (as produced by PromptBuilder.build_contents) into the
        str / PIL.Image parts the SDK expects."""

        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, np.ndarray):
                rgb = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
                parts.append(Image.fromarray(rgb))
            else:
                raise TypeError(f"Unsupported content part type: {type(item)}")
        return parts

    @staticmethod
    def _extract_json(text: str):
        """Parses the model's response text as a JSON array.
        Defensively strips ```json ... ``` fences even though the
        request asked for response_mime_type="application/json" —
        some responses still wrap output in fences under load.
        Returns None (never raises) so the caller can decide
        whether to retry."""

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return data if isinstance(data, list) else None

    def _backoff(self, attempt: int) -> float:
        base = min(5 * (attempt + 1), 30)
        return base + random.uniform(0, 1.5)

    # =====================================================
    # Generate Explanation
    # =====================================================

    def generate(self, contents: list) -> list:
        """contents: ordered list of str / BGR numpy arrays, e.g.
        from PromptBuilder.build_contents(). Returns the parsed
        JSON array of per-defect explanation dicts.

        Raises RuntimeError if every attempt fails — callers
        (ExplanationEngine) are expected to catch this and fall
        back to services.defect_knowledge rather than losing the
        whole inspection result over an LLM hiccup.
        """

        parts = self._to_parts(contents)
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            except APIError as e:
                last_error = e
                code = getattr(e, "code", None)
                print(f"Gemini API error {code} on attempt {attempt + 1}/{self.max_attempts}: {e}")
                if attempt < self.max_attempts - 1:
                    time.sleep(self._backoff(attempt))
                    continue
                raise RuntimeError(f"Gemini API error after {self.max_attempts} attempts: {e}") from e

            text = getattr(response, "text", None)
            if not text:
                # Empty text commonly means the response was blocked
                # by a safety filter, or the model returned only a
                # function-call/thought part with no text part.
                last_error = RuntimeError(
                    "Gemini returned an empty response (possibly filtered)."
                )
                print(f"Empty Gemini response on attempt {attempt + 1}/{self.max_attempts}")
                if attempt < self.max_attempts - 1:
                    time.sleep(self._backoff(attempt))
                    continue
                raise last_error

            data = self._extract_json(text)
            if data is not None:
                return data

            last_error = ValueError("Gemini did not return a valid JSON array.")
            print(f"Invalid JSON on attempt {attempt + 1}/{self.max_attempts}:\n{text[:500]}")
            if attempt < self.max_attempts - 1:
                time.sleep(self._backoff(attempt))
                continue

        raise RuntimeError("Gemini Service is currently unavailable.") from last_error
