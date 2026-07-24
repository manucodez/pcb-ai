"""
=========================================================
PCB Explanation Engine

Purpose
-------
Generate AI explanations for all detected PCB defects.

Input
-----
Original PCB Image
YOLO Detection JSON

Output
------
Detection JSON with AI explanations

Author : Jasmeen
=========================================================
"""

from services.crop_defects import DefectCropper
from services.prompt_builder import PromptBuilder
from services.gemini_service import GeminiService


class ExplanationEngine:

    def __init__(self):

        self.cropper = DefectCropper()

        self.prompt_builder = PromptBuilder()

        self.gemini = GeminiService()

    # =====================================================
    # Generate Explanations
    # =====================================================

    def generate(self, image_path, detections):

        # -------------------------------------------------
        # Crop all detected defects
        # -------------------------------------------------

        crops = self.cropper.crop_defects(
            image_path=image_path,
            detections=detections
        )

        if not crops:
            return []

        # -------------------------------------------------
        # Build Prompt
        # -------------------------------------------------

        prompt = self.prompt_builder.build_prompt(
            crops
        )

        # -------------------------------------------------
        # Collect Images
        # -------------------------------------------------

        images = [
            crop["image"]
            for crop in crops
        ]

        # -------------------------------------------------
        # Gemini
        # -------------------------------------------------

        explanations = self.gemini.generate(
            images=images,
            prompt=prompt
        )

        # =================================================
        # Validation
        # =================================================

        if len(explanations) != len(crops):

            raise RuntimeError(

                "Gemini returned "
                f"{len(explanations)} explanations "

                f"for {len(crops)} defects."

            )

        expected_ids = {

            crop["id"]

            for crop in crops

        }

        received_ids = {

            item["id"]

            for item in explanations

        }

        if expected_ids != received_ids:

            raise RuntimeError(

                "Mismatch between "
                "YOLO defect IDs and "
                "Gemini response IDs."

            )

        # =================================================
        # Create Lookup Dictionary
        # =================================================

        explanation_map = {

            item["id"]: item

            for item in explanations

        }

        # =================================================
        # Merge YOLO + Gemini
        # =================================================

        results = []

        for crop in crops:

            explanation = explanation_map[crop["id"]]

            # Remove duplicate id
            explanation.pop("id", None)

            results.append(

                {

                    "id": crop["id"],

                    "class": crop["class"],

                    "confidence": crop["confidence"],

                    "bbox": crop["bbox"],

                    "explanation": explanation

                }

            )

        return results