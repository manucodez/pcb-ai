"""
=========================================================
PCB Prompt Builder

Purpose
-------
Build the Gemini prompt content for a batch of detected
defects.

Design note — why interleaved, not one text block + N images
--------------------------------------------------------------
The previous version sent one long text block describing every
defect ID in order, followed by all N crop images back to back.
That leaves the image-to-defect correspondence entirely
implicit in ordering, which gets easier for a vision-language
model to lose track of as the defect count grows.

Instead, build_contents() below interleaves a short label
directly before each image: "--- Defect 3 ---", metadata, a
domain hint, *then* that defect's image. Every image is
immediately preceded by an unambiguous statement of which
defect it is, which is the same technique used for grounding
multi-image prompts in general.

Input
-----
List of Detected Defects (each a dict from DefectCropper,
without the "image" key removed — build_contents needs it to
build the final ordered content list).

Output
------
An ordered list mixing plain strings (text parts) and BGR
numpy image arrays — ready for GeminiService to convert and
send as-is.

Author : Manjeet
=========================================================
"""

from __future__ import annotations

from services.defect_knowledge import lookup

INTRO = """You are an expert PCB Quality Inspection Engineer reviewing \
automated defect detections from a YOLO model.

You will receive several cropped PCB defect images, each preceded by a \
short label block identifying its Defect ID, the YOLO-predicted class, \
detector confidence, and a brief domain hint for that defect type. Use \
the label block, the domain hint, AND what you actually see in the \
image together — the YOLO class is a starting hypothesis, not ground \
truth, so if the image clearly shows something else, correct it in \
"defect_type" and explain why in "explanation".

For every defect image:
1. Verify (or correct) the detected defect type using the image.
2. Explain what is visibly wrong.
3. State the likely manufacturing root cause.
4. Estimate severity.
5. Suggest a concrete repair.
6. Suggest a preventive measure for the fab process.

Return ONLY valid JSON — a single JSON array, one object per defect, \
in this exact shape:

[
    {
        "id": 1,
        "defect_type": "",
        "explanation": "",
        "root_cause": "",
        "severity": "",
        "recommended_fix": "",
        "prevention": ""
    }
]

Rules:
1. Return ONLY JSON. No markdown, no code fences, no prose outside the array.
2. The number of JSON objects MUST equal the number of defect images you were shown.
3. Each "id" MUST exactly match the Defect ID given in that image's label block — do not renumber.
4. Keep explanations concise (1-3 sentences per field).
5. The "severity" field MUST be exactly one of these four strings, spelled \
and capitalized exactly as shown: "Critical", "High", "Medium", "Low". Do \
not use any other word, synonym, or casing.

The defects, in the order their images will follow, are:
"""


class PromptBuilder:

    def build_contents(self, defects: list[dict]) -> list:
        """defects: list of dicts with at least id, class,
        confidence, image (BGR numpy array) — as produced by
        DefectCropper.crop_defects().

        Returns an ordered list of str / numpy.ndarray items
        ready to hand to GeminiService.generate(contents=...).
        """

        contents: list = [INTRO]

        for defect in defects:
            hint = lookup(defect["class"]).get("prompt_hint", "")
            label = (
                f"\n--- Defect ID {defect['id']} ---\n"
                f"YOLO class      : {defect['class']}\n"
                f"Detector conf.  : {defect['confidence']:.2f}\n"
                f"Domain hint     : {hint}\n"
                f"(the image for Defect ID {defect['id']} follows immediately below)\n"
            )
            contents.append(label)
            contents.append(defect["image"])

        return contents

    # Kept for backward compatibility with any external caller that
    # only wants the text portion (e.g. logging/debugging the prompt
    # without images).
    def build_prompt(self, defects: list[dict]) -> str:
        parts = [p for p in self.build_contents(defects) if isinstance(p, str)]
        return "".join(parts)
