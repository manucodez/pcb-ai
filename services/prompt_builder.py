"""
=========================================================
PCB Prompt Builder

Purpose
-------
Build a single prompt for all detected PCB defects.

Input
-----
List of Detected Defects

Output
------
Prompt String

Author : Jasmeen
=========================================================
"""


class PromptBuilder:

    def build_prompt(self, defects):

        prompt = """
You are an expert PCB Quality Inspection Engineer.

You will receive multiple cropped PCB defect images.

Each image corresponds to one detected defect.

For every image:

1. Verify the detected defect using the image.
2. Explain the defect.
3. Mention the likely manufacturing root cause.
4. Estimate severity.
5. Suggest a repair.
6. Suggest preventive measures.

Use BOTH:
- the cropped defect image
- the YOLO detection information

The detections are:

"""

        for defect in defects:

            prompt += f"""

Defect ID : {defect['id']}
YOLO Class : {defect['class']}
Confidence : {defect['confidence']:.2f}

"""

        prompt += """

Return ONLY valid JSON.

Return a JSON ARRAY.

Example:

[
    {
        "id":1,
        "defect_type":"",
        "explanation":"",
        "root_cause":"",
        "severity":"",
        "recommended_fix":"",
        "prevention":""
    }
]

Rules

1. Return ONLY JSON.
2. No markdown.
3. No code blocks.
4. Do not explain outside JSON.
5. The number of JSON objects MUST equal the number of images.
6. Keep explanations concise.
"""

        return prompt