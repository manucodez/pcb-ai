"""
=========================================================
Gemini Service

Purpose
-------
Generate AI explanations for PCB defects
using Google Gemini Vision.

Input
-----
List of Cropped Images
Prompt

Output
------
List of AI Explanations

Author : Jasmeen
=========================================================
"""

import json
import os
import time

import cv2
from PIL import Image

from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.errors import ServerError


load_dotenv()


class GeminiService:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:

            raise ValueError(
                "Gemini API Key not found."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.5-flash"

        print(f"Using Gemini Model : {self.model}")

    # =====================================================
    # Generate Explanation
    # =====================================================

    def generate(self, images, prompt):

        contents = [prompt]

        # -----------------------------------------------
        # Convert OpenCV → PIL
        # -----------------------------------------------

        for image in images:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(image)

            contents.append(image)

        # -----------------------------------------------
        # Retry if Gemini Busy
        # -----------------------------------------------

        max_attempts = 3

        for attempt in range(max_attempts):

            try:

                response = self.client.models.generate_content(

                    model=self.model,

                    contents=contents,

                    config=types.GenerateContentConfig(

                        response_mime_type="application/json"

                    )

                )

                text = response.text.strip()

                data = json.loads(text)

                if not isinstance(data, list):

                    raise RuntimeError(
                        "Gemini returned non-list JSON."
                    )

                return data

            except json.JSONDecodeError:

                print("\nInvalid JSON Returned\n")

                print(response.text)

                raise

            except ServerError as e:

                wait = min(5 * (attempt + 1), 30)

                print(
        f"Gemini Server Busy (503)"
        f"\nRetry {attempt + 1}/{max_attempts}"
        f"\nWaiting {wait} seconds..."
    )

                time.sleep(wait)

        raise RuntimeError(

            "Gemini Service is currently unavailable."

        )