"""
=========================================================
PCB Defect Domain Knowledge

Purpose
-------
A small, static knowledge base describing the six standard
PCB copper-defect classes this project trains on:

    missing_hole, mouse_bite, open_circuit,
    short, spur, spurious_copper

Used in two places:

1. ExplanationEngine — as a deterministic FALLBACK when the
   Gemini call fails (rate limit, outage, malformed output)
   after retries, so a defect never silently disappears from
   a report just because the LLM call had a bad moment. The
   YOLO detection is always real; only the natural-language
   explanation degrades.

2. PromptBuilder — as a short domain hint injected next to
   each defect so Gemini starts from the correct manufacturing
   vocabulary instead of guessing, which measurably reduces
   drift on the "root_cause" and "severity" fields.

Keep this file dependency-free (no cv2/torch/genai) so it can
be imported and unit-tested in isolation.
=========================================================
"""

from __future__ import annotations

# Baseline severity if a class is detected but nothing else is
# known — used only as the last-resort default.
DEFAULT_SEVERITY = "Medium"

DEFECT_KB: dict[str, dict[str, str]] = {
    "missing_hole": {
        "severity": "High",
        "explanation": (
            "A drilled through-hole (via or component lead hole) that the "
            "design calls for is absent or malformed at this location."
        ),
        "root_cause": (
            "Drill program / Gerber-to-NC-drill mismatch, panel "
            "misregistration during drilling, or an undetected broken "
            "drill bit."
        ),
        "recommended_fix": (
            "If caught pre-assembly: re-drill and re-plate the hole. If "
            "the board is already populated, this typically requires a "
            "jumper wire or component-level rework, or scrapping the "
            "board depending on the net's criticality."
        ),
        "prevention": (
            "Cross-check the drill file against the latest Gerbers before "
            "fabrication, monitor drill-bit wear, and run automated "
            "optical inspection immediately after the drilling step."
        ),
        "prompt_hint": (
            "expected hole absent or malformed; usually a drill-program "
            "or registration error"
        ),
    },
    "mouse_bite": {
        "severity": "Medium",
        "explanation": (
            "A small semi-circular notch is bitten out of a copper trace "
            "or pad edge, narrowing the conductor at that point."
        ),
        "root_cause": (
            "Over-etching, mask misalignment, or mechanical stress during "
            "panel depaneling (routing/punching)."
        ),
        "recommended_fix": (
            "If the remaining trace width still meets IPC clearance/"
            "current-carrying requirements it can pass with a note; "
            "otherwise repair with conductive bridging or scrap the board."
        ),
        "prevention": (
            "Tighten etch time/chemistry control, improve mask "
            "registration, and prefer V-scoring or routing over punch "
            "depaneling for tight-tolerance boards."
        ),
        "prompt_hint": (
            "notch bitten out of a trace/pad edge; check remaining "
            "conductor width"
        ),
    },
    "open_circuit": {
        "severity": "Critical",
        "explanation": (
            "A break interrupts a copper trace that should be "
            "electrically continuous, breaking that net."
        ),
        "root_cause": (
            "Under-etching leftover break, physical trace damage, foil "
            "defect, or contamination introduced during lamination."
        ),
        "recommended_fix": (
            "Bridge the break with conductive epoxy or a jumper wire for "
            "low-volume rework; for production, scrap and correct the "
            "fab process, and re-run electrical test on the lot."
        ),
        "prevention": (
            "Add 100% automated electrical continuity (bare-board) "
            "testing after etch, and tighten etch process control."
        ),
        "prompt_hint": (
            "trace is broken / not continuous; this is a functional "
            "failure of the net, treat as high-severity unless the crop "
            "shows the trace is in fact intact"
        ),
    },
    "short": {
        "severity": "Critical",
        "explanation": (
            "Unwanted copper bridges two traces or pads that are supposed "
            "to be electrically isolated."
        ),
        "root_cause": (
            "Under-etching (residual copper not fully removed), a "
            "photoresist defect, or solder bridging introduced during "
            "assembly."
        ),
        "recommended_fix": (
            "Manually remove the bridging copper (scrape or etch "
            "touch-up), then re-run electrical test before the board "
            "ships."
        ),
        "prevention": (
            "Tighten photoresist exposure/development and etch process "
            "windows, and add AOI plus electrical test as a gate."
        ),
        "prompt_hint": (
            "extra copper bridges two nets that should be isolated; "
            "usually critical"
        ),
    },
    "spur": {
        "severity": "Low",
        "explanation": (
            "A thin, unwanted finger of copper projects from a trace or "
            "pad where the design shows a clean edge."
        ),
        "root_cause": (
            "Incomplete etching, a small photoresist/mask defect, or an "
            "artwork generation error."
        ),
        "recommended_fix": (
            "Usually trimmed away during routine touch-up inspection; "
            "escalate only if it approaches an adjacent net."
        ),
        "prevention": (
            "Etch/mask process control and a design-for-manufacturing "
            "clearance check on the artwork before release."
        ),
        "prompt_hint": (
            "thin unwanted copper finger off a trace/pad; severity "
            "depends on how close it comes to a neighboring net"
        ),
    },
    "spurious_copper": {
        "severity": "Low",
        "explanation": (
            "An unwanted island or patch of copper appears on the board "
            "with no connection to the intended circuit pattern."
        ),
        "root_cause": (
            "A pinhole or defect in the etch resist that let copper "
            "survive where it should have been removed, or artwork/"
            "contamination issues."
        ),
        "recommended_fix": (
            "Manual removal or etch touch-up; verify with electrical "
            "test if the island sits close to an active net."
        ),
        "prevention": (
            "Improve photoresist application cleanliness and QC on the "
            "generated artwork before fabrication."
        ),
        "prompt_hint": (
            "isolated island of copper unrelated to the circuit pattern; "
            "escalate if near a high-density net"
        ),
    },
}


def lookup(class_name: str) -> dict[str, str]:
    """Case/format-tolerant lookup. Falls back to a generic
    entry for classes not in the KB (e.g. a future retrain adds
    a class this file hasn't been updated for yet) instead of
    raising, since a fallback lookup must never itself fail."""

    key = (class_name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in DEFECT_KB:
        return DEFECT_KB[key]

    return {
        "severity": DEFAULT_SEVERITY,
        "explanation": (
            f"A '{class_name}' defect was detected by the model. No "
            "domain reference entry exists yet for this class."
        ),
        "root_cause": "Not available for this defect class.",
        "recommended_fix": (
            "Have a human inspector review this detection directly."
        ),
        "prevention": "Not available for this defect class.",
        "prompt_hint": "no domain reference available for this class",
    }


def fallback_explanation(class_name: str) -> dict[str, str]:
    """Build a full explanation dict in the same shape Gemini
    would return, used when the AI stage is unavailable. Tagged
    with ai_generated=False so the UI can show it was not
    reviewed by the vision model."""

    kb = lookup(class_name)
    return {
        "defect_type": class_name,
        "explanation": kb["explanation"],
        "root_cause": kb["root_cause"],
        "severity": kb["severity"],
        "recommended_fix": kb["recommended_fix"],
        "prevention": kb["prevention"],
        "ai_generated": False,
    }
