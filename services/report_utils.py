"""
=========================================================
PCB Report Utilities

Purpose
-------
Pure, Streamlit-free logic shared by the UI layer:
severity normalization, verdict/sorting logic, and the
Markdown report builder.

Why this file exists
---------------------
This logic used to live inline in app.py. That made it
untestable without booting a full Streamlit session (importing
app.py runs st.set_page_config() at import time) and meant the
same severity rules could drift from what the annotator draws.
Extracting it here means:

  - app.py and any future UI (CLI, API) share one source of
    truth for severity handling.
  - tests/test_report_utils.py can exercise it directly.

Author : Manjeet
=========================================================
"""

from __future__ import annotations

from collections import Counter, defaultdict

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITY_ORDER)}
SEVERITY_RANK["Unknown"] = len(SEVERITY_ORDER)

# Synonyms the model might drift into despite the prompt's constraint —
# normalized defensively so the UI never silently breaks on an
# unexpected string.
_SEVERITY_SYNONYMS = {
    "critical": "Critical", "severe": "Critical", "blocker": "Critical",
    "high": "High", "major": "High",
    "medium": "Medium", "moderate": "Medium",
    "low": "Low", "minor": "Low", "cosmetic": "Low",
}


def normalize_severity(raw) -> str:
    if not raw:
        return "Unknown"
    key = str(raw).strip().lower()
    if key in ("critical", "high", "medium", "low"):
        return key.capitalize()
    return _SEVERITY_SYNONYMS.get(key, "Unknown")


def normalize_result(result: dict) -> dict:
    for det in result.get("detections", []):
        exp = det.get("explanation") or {}
        exp["severity"] = normalize_severity(exp.get("severity"))
        det["explanation"] = exp
    return result


def overall_verdict(detections: list) -> tuple[str, str, str]:
    """Returns (label, color, message)."""
    if not detections:
        return "PASS", "#2e8b57", "No defects detected on this board."

    severities = {d.get("explanation", {}).get("severity", "Unknown") for d in detections}

    if "Critical" in severities:
        return "FAIL", "#c0392b", "Critical defect(s) found — do not release without rework."
    if "High" in severities:
        return "HOLD — REVIEW REQUIRED", "#d9822b", "High-severity defect(s) found — inspect before release."
    if "Medium" in severities:
        return "HOLD — REVIEW REQUIRED", "#b8960c", "Medium-severity defect(s) found — recommended rework."
    return "PASS — MINOR ISSUES", "#2e8b57", "Only low-severity / cosmetic defects found."


def sorted_by_severity(detections: list) -> list:
    return sorted(
        detections,
        key=lambda d: SEVERITY_RANK.get(d.get("explanation", {}).get("severity", "Unknown"), 99),
    )


def dedupe_labels(names: list) -> list:
    """Turns repeated filenames into unique, still-readable labels
    (e.g. two uploads named board.jpg -> 'board.jpg' and 'board.jpg (2)')."""
    counts = Counter(names)
    seen = defaultdict(int)
    labels = []
    for name in names:
        seen[name] += 1
        labels.append(name if counts[name] == 1 else f"{name} ({seen[name]})")
    return labels


def build_markdown_report(result: dict) -> str:
    label, _, message = overall_verdict(result["detections"])
    settings = result.get("scan_settings", {})

    lines = [
        f"# PCB Inspection Report — {result['image']}",
        "",
        f"**Verdict:** {label} — {message}",
        f"**Total defects:** {result['total_defects']}",
        f"**Processing time:** {result.get('pipeline_time_sec', '—')}s",
        f"**Scan settings:** {settings.get('imgsz', '—')}px"
        + (" · thorough (multi-view)" if settings.get("augment") else " · balanced")
        + (" · tiled" if settings.get("tiled") else "")
        + f" · confidence ≥ {settings.get('confidence', '—')} · IoU {settings.get('iou', '—')}",
        "",
    ]

    for det in sorted_by_severity(result["detections"]):
        exp = det.get("explanation", {})
        ai_note = "" if exp.get("ai_generated", True) else " _(standard guidance — AI review unavailable for this defect)_"
        lines += [
            f"## Defect #{det.get('id', '?')} — {det['class'].replace('_', ' ').title()}{ai_note}",
            f"- **Severity:** {exp.get('severity', 'N/A')}",
            f"- **Confidence:** {det['confidence']:.1%}",
            f"- **What's wrong:** {exp.get('explanation', 'N/A')}",
            f"- **Likely cause:** {exp.get('root_cause', 'N/A')}",
            f"- **Recommended fix:** {exp.get('recommended_fix', 'N/A')}",
            f"- **Prevention:** {exp.get('prevention', 'N/A')}",
            "",
        ]

    if not result["detections"]:
        lines.append("_No defects detected — board passed inspection._")

    return "\n".join(lines)
