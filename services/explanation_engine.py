"""
=========================================================
PCB Explanation Engine

Purpose
-------
Generate AI explanations for all detected PCB defects.

Reliability note
------------------
The previous version raised on ANY problem with Gemini's
response — a wrong item count, one bad/missing id, a full API
outage — which discarded every valid YOLO detection for that
image along with it. A user re-uploading the same board could
get a completely empty report just because the LLM had a
formatting hiccup on one of five defects.

This version treats the Gemini explanation as an enhancement
layer, not a dependency the whole result lives or dies on:
  - Per-item validation keeps whatever Gemini got right and
    only falls back (services.defect_knowledge) for the
    specific defect(s) it got wrong or omitted.
  - A total Gemini failure (auth error, outage, repeated bad
    JSON) falls back for every defect instead of raising —
    the YOLO detections and bounding boxes are always real and
    are never thrown away because of the explanation stage.

Every result item carries explanation["ai_generated"] so the UI
can show which explanations were actually reviewed by Gemini
vs. standard reference guidance.

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

from __future__ import annotations

from services.crop_defects import DefectCropper
from services.prompt_builder import PromptBuilder
from services.gemini_service import GeminiService
from services.defect_knowledge import fallback_explanation

REQUIRED_FIELDS = [
    "defect_type", "explanation", "root_cause",
    "severity", "recommended_fix", "prevention",
]


class ExplanationEngine:

    def __init__(self):
        self.cropper = DefectCropper()
        self.prompt_builder = PromptBuilder()
        self.gemini = GeminiService()

    # =====================================================
    # Generate Explanations
    # =====================================================

    def generate(self, image_path, detections):

        crops = self.cropper.crop_defects(image_path=image_path, detections=detections)

        if not crops:
            return []

        explanation_map = self._get_ai_explanations(crops)

        results = []
        for crop in crops:
            explanation = explanation_map.get(crop["id"]) or fallback_explanation(crop["class"])
            results.append({
                "id": crop["id"],
                "class": crop["class"],
                "confidence": crop["confidence"],
                "bbox": crop["bbox"],
                "explanation": explanation,
            })

        return results

    # =====================================================
    # Gemini call + validation (never raises)
    # =====================================================

    def _get_ai_explanations(self, crops: list[dict]) -> dict[int, dict]:
        try:
            contents = self.prompt_builder.build_contents(crops)
            raw = self.gemini.generate(contents=contents)
        except Exception as e:
            print(f"Gemini explanation stage unavailable, using fallback guidance for all defects: {e}")
            return {}

        return self._validate_and_index(raw, crops)

    def _validate_and_index(self, raw, crops: list[dict]) -> dict[int, dict]:
        """Best-effort validation: keep whatever items have a
        valid, unique id matching a real crop and backfill any
        missing field with 'N/A' rather than discarding the whole
        response over one malformed item. Anything that can't be
        matched to a real defect id is silently ignored — that
        defect gets fallback guidance instead."""

        valid_ids = {c["id"] for c in crops}
        indexed: dict[int, dict] = {}

        if not isinstance(raw, list):
            return indexed

        for item in raw:
            if not isinstance(item, dict):
                continue

            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue

            if item_id not in valid_ids or item_id in indexed:
                continue

            explanation = {field: item.get(field) or "N/A" for field in REQUIRED_FIELDS}
            explanation["ai_generated"] = True
            indexed[item_id] = explanation

        missing = valid_ids - indexed.keys()
        if missing:
            print(
                f"Gemini response missing or invalid for defect id(s) "
                f"{sorted(missing)} — using standard guidance for those."
            )

        return indexed
