"""
=========================================================
PCB Defect Inspection Tool - Streamlit UI

Purpose
-------
Upload one or more PCB images, run each through the full
inspection pipeline (YOLO detection + Gemini explanation),
and display an annotated image plus a defect-by-defect
report in plain language, with previewable + downloadable
JSON and Markdown reports.

Run with:
    streamlit run app.py

Author : Manjeet
=========================================================
"""

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from services.pipeline import PCBInspectionPipeline
from services.crop_defects import enhance_clarity
from services.image_quality import assess_quality
from services.report_utils import (
    SEVERITY_ORDER,
    normalize_result,
    overall_verdict,
    sorted_by_severity,
    dedupe_labels,
    build_markdown_report,
)
from training.inference import DEFAULT_MODEL_PATH, MODELS_DIR, RUNS_DIR

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🛠️",
    layout="wide",
)

APP_TITLE = "PCB Defect Inspector"
APP_TAGLINE = "AI-assisted visual inspection for printed circuit boards"

SEVERITY_STYLE = {
    "Critical": {"bg": "#c0392b", "icon": "⛔"},
    "High":     {"bg": "#d9822b", "icon": "🔶"},
    "Medium":   {"bg": "#b8960c", "icon": "🟡"},
    "Low":      {"bg": "#2ecc71", "icon": "🟢"},
    "Unknown":  {"bg": "#7f8c8d", "icon": "⚪"},
}

BOX_COLOR_BGR = {
    "Critical": (30, 30, 192),
    "High": (30, 130, 217),
    "Medium": (12, 150, 184),
    "Low": (87, 190, 60),
    "Unknown": (150, 150, 150),
}

# Detection modes: (imgsz, augment/TTA, tiled sliding-window scan, description)
DETECTION_MODES = {
    "Balanced": {"imgsz": 1280, "augment": False, "tiled": False},
    "Thorough (slower, higher recall)": {"imgsz": 1536, "augment": True, "tiled": False},
    "Maximum recall — tiled scan (slowest)": {"imgsz": 1280, "augment": False, "tiled": True},
}

MODE_HELP = (
    "Balanced: fast, higher-resolution scan than the model's training "
    "size — good default.\n\n"
    "Thorough: adds multi-view test-time augmentation for better recall "
    "on faint or small defects, at roughly 2-3x the processing time.\n\n"
    "Maximum recall (tiled scan): splits high-resolution images into "
    "overlapping tiles and scans each at full detail before merging "
    "results — the best option for catching tiny defects (mouse bites, "
    "spurs, pin-holes) on large board photos, at the cost of the "
    "slowest processing time."
)


# =========================================================
# Session-scoped scratch space
# =========================================================

if "_session_id" not in st.session_state:
    st.session_state["_session_id"] = str(uuid.uuid4())

SESSION_DIR = Path(tempfile.gettempdir()) / "pcb_ai_sessions" / st.session_state["_session_id"]
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Styling
# =========================================================

def inject_css():
    st.markdown(
        """
        <style>
        .pcb-header-wrap {
            padding-bottom: 10px;
            border-bottom: 1px solid #22303c;
            margin-bottom: 20px;
        }
        .pcb-header-title {
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.2;
            background: linear-gradient(90deg, #3fa7ff, #7ee8e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .pcb-header-tagline {
            color: #8a95a1;
            font-size: 1rem;
            margin-top: 2px;
        }
        .pcb-badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            color: white;
            text-transform: uppercase;
        }
        .pcb-verdict {
            padding: 14px 20px;
            border-radius: 10px;
            font-size: 1.05rem;
            font-weight: 650;
            color: white;
            margin-bottom: 14px;
        }
        .pcb-verdict-sub {
            font-weight: 400;
            font-size: 0.92rem;
            opacity: 0.92;
        }
        .pcb-footer {
            color: #8a95a1;
            font-size: 0.82rem;
            text-align: center;
            margin-top: 2.5rem;
        }
        .pcb-fallback-note {
            color: #8a95a1;
            font-size: 0.8rem;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_badge_html(severity: str) -> str:
    style = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["Unknown"])
    return (
        f"<span class='pcb-badge' style='background-color:{style['bg']}'>"
        f"{style['icon']} {severity}</span>"
    )


# =========================================================
# Environment / readiness checks
# =========================================================

def check_environment() -> list[str]:
    problems = []

    if not os.getenv("GEMINI_API_KEY"):
        problems.append(
            "**GEMINI_API_KEY** is not set. Add it to a `.env` file in the "
            "project root (`GEMINI_API_KEY=your_key_here`) or set it as an "
            "environment variable, then restart the app."
        )

    if not DEFAULT_MODEL_PATH.exists():
        problems.append(
            f"No trained model found. Checked `{MODELS_DIR / 'best.pt'}` and "
            f"`{RUNS_DIR}/*/weights/best.pt`. Train one with "
            "`python training/train_yolo.py` (auto-promotes the result), or "
            "point the `PCB_MODEL_PATH` environment variable at an existing "
            "`.pt` file."
        )

    return problems


@st.cache_resource(show_spinner="Loading detection model…")
def load_pipeline():
    return PCBInspectionPipeline()


# =========================================================
# Cached inspection — avoids re-calling the model/Gemini
# every time an unrelated widget triggers a Streamlit rerun.
# The leading underscore on `_pipeline` tells Streamlit to
# skip hashing that argument (it isn't hashable anyway).
# =========================================================

@st.cache_data(show_spinner=False, max_entries=100)
def run_inspection_cached(
    _pipeline, image_bytes: bytes, dest_name: str,
    confidence: float, iou: float, imgsz: int, augment: bool, tiled: bool,
):
    dest = SESSION_DIR / dest_name
    dest.write_bytes(image_bytes)
    result = _pipeline.inspect(
        dest, confidence=confidence, iou=iou, imgsz=imgsz, augment=augment, tiled=tiled,
    )
    return normalize_result(result)


# =========================================================
# Image helpers — image is read from disk once per file,
# not once per defect card. Thumbnails go through the same
# clarity enhancement the backend uses on Gemini's crops, so
# what the user sees matches what the model actually analyzed.
# =========================================================

def load_image_bgr(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def draw_boxes(img_bgr, detections: list):
    img = img_bgr.copy()
    for det in detections:
        bbox = det["bbox"]
        severity = det.get("explanation", {}).get("severity", "Unknown")
        color = BOX_COLOR_BGR.get(severity, (150, 150, 150))

        x1, y1, x2, y2 = (int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{det.get('id', '?')} {det['class']}"
        cv2.putText(img, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_thumbnail(img_bgr, bbox: dict, padding: int = 15):
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(bbox["x1"]) - padding)
    y1 = max(0, int(bbox["y1"]) - padding)
    x2 = min(w, int(bbox["x2"]) + padding)
    y2 = min(h, int(bbox["y2"]) + padding)
    crop = img_bgr[y1:y2, x1:x2]
    crop = enhance_clarity(crop)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# =========================================================
# Rendering
# =========================================================

def render_defect_card(det: dict, img_bgr):
    exp = det.get("explanation", {})
    severity = exp.get("severity", "Unknown")

    with st.container(border=True):
        thumb_col, info_col = st.columns([1, 3])

        with thumb_col:
            try:
                st.image(crop_thumbnail(img_bgr, det["bbox"]), use_container_width=True)
            except Exception:
                st.caption("Preview unavailable")

        with info_col:
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"**#{det.get('id', '?')} — {det['class'].replace('_', ' ').title()}**")
            with top_right:
                st.markdown(severity_badge_html(severity), unsafe_allow_html=True)

            st.caption(f"Detection confidence: {det['confidence']:.1%}")
            st.markdown(f"**What's wrong**  \n{exp.get('explanation', 'N/A')}")
            st.markdown(f"**Likely cause**  \n{exp.get('root_cause', 'N/A')}")
            st.markdown(f"**Recommended fix**  \n{exp.get('recommended_fix', 'N/A')}")
            st.markdown(f"**Prevention**  \n{exp.get('prevention', 'N/A')}")

            if not exp.get("ai_generated", True):
                st.markdown(
                    "<span class='pcb-fallback-note'>ℹ️ Standard reference guidance — "
                    "AI review was unavailable for this specific defect.</span>",
                    unsafe_allow_html=True,
                )


def render_single_result(image_path: Path, result: dict, quality: dict, key_suffix: str):
    detections = sorted_by_severity(result.get("detections", []))
    total = result.get("total_defects", 0)
    elapsed = result.get("pipeline_time_sec", None)
    settings = result.get("scan_settings", {})
    img_bgr = load_image_bgr(image_path)

    for warning in quality.get("warnings", []):
        st.warning(warning, icon="⚠️")

    verdict_label, verdict_color, verdict_msg = overall_verdict(detections)
    st.markdown(
        f"<div class='pcb-verdict' style='background-color:{verdict_color}'>"
        f"{verdict_label}<div class='pcb-verdict-sub'>{verdict_msg}</div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Annotated image**")
        if detections:
            st.image(draw_boxes(img_bgr, detections), use_container_width=True)
        else:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        if settings:
            if settings.get("tiled"):
                mode_note = "maximum recall (tiled scan)"
            elif settings.get("augment"):
                mode_note = "thorough (multi-view)"
            else:
                mode_note = "balanced"
            st.caption(f"Scanned at {settings.get('imgsz')}px · {mode_note}")

    with right:
        st.markdown("**Summary**")
        m1, m2 = st.columns(2)
        m1.metric("Defects found", total)
        if elapsed is not None:
            m2.metric("Processed in", f"{elapsed}s")

        if total:
            st.write("")
            for sev in SEVERITY_ORDER:
                count = sum(1 for d in detections if d.get("explanation", {}).get("severity") == sev)
                if count:
                    st.markdown(f"{severity_badge_html(sev)} &nbsp; {count}", unsafe_allow_html=True)

        st.write("")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "JSON report",
                data=json.dumps(result, indent=2),
                file_name=f"{Path(result['image']).stem}_report.json",
                mime="application/json",
                key=f"json_dl_{key_suffix}",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Markdown report",
                data=build_markdown_report(result),
                file_name=f"{Path(result['image']).stem}_report.md",
                mime="text/markdown",
                key=f"md_dl_{key_suffix}",
                use_container_width=True,
            )

    if not detections:
        return

    st.write("")
    tab_details, tab_markdown, tab_json = st.tabs(
        ["Defect details", "Markdown report", "Raw JSON"]
    )

    with tab_details:
        st.caption("Sorted by severity — most critical first.")
        for det in detections:
            render_defect_card(det, img_bgr)

    with tab_markdown:
        with st.container(border=True):
            st.markdown(build_markdown_report(result))

    with tab_json:
        with st.container(border=True):
            st.code(json.dumps(result, indent=2), language="json")


# =========================================================
# Main app
# =========================================================

def main():
    inject_css()

    st.markdown(
        f"<div class='pcb-header-wrap'>"
        f"<div class='pcb-header-title'>🛠️ {APP_TITLE}</div>"
        f"<div class='pcb-header-tagline'>{APP_TAGLINE}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    problems = check_environment()
    if problems:
        st.error(
            "**Setup incomplete** — fix the following before continuing:\n\n"
            + "\n\n".join(f"- {p}" for p in problems)
        )
        return

    pipeline = load_pipeline()

    with st.sidebar:
        st.markdown("### Detection mode")
        mode = st.radio(
            "Scan thoroughness",
            list(DETECTION_MODES.keys()),
            help=MODE_HELP,
            label_visibility="collapsed",
        )
        mode_settings = DETECTION_MODES[mode]

        st.divider()
        st.markdown("### Inspection settings")
        confidence = st.slider(
            "Confidence threshold", 0.05, 0.95, 0.50, 0.05,
            help="Minimum detection confidence for a defect to be reported. Lower catches more possible defects but increases false positives.",
        )
        iou = st.slider(
            "IoU threshold", 0.05, 0.95, 0.40, 0.05,
            help="Overlap threshold used to merge duplicate detections of the same defect.",
        )

        st.divider()
        with st.expander("About this model"):
            try:
                class_names = sorted(pipeline.detector.model.names.values())
                st.caption("Defect classes this model can detect:")
                for c in class_names:
                    st.markdown(f"- {c.replace('_', ' ').title()}")
            except Exception:
                st.caption("Class list unavailable.")
            st.caption("Detection: YOLOv11 · Explanations: Gemini Vision")
            st.caption(f"Weights: `{pipeline.detector.model_path}`")

        st.divider()
        if st.button("Clear session data", use_container_width=True):
            run_inspection_cached.clear()
            shutil.rmtree(SESSION_DIR, ignore_errors=True)
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload PCB image(s) for inspection",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        help="You can select multiple boards at once — each gets its own report tab.",
    )

    if not uploaded_files:
        st.info("Upload one or more board images to begin inspection.")
        return

    labels = dedupe_labels([f.name for f in uploaded_files])

    results = []  # list of (label, image_path, result, quality)
    progress = st.progress(0.0, text="Starting inspection…")

    for i, (uploaded_file, label) in enumerate(zip(uploaded_files, labels)):
        progress.progress(
            i / len(uploaded_files),
            text=f"Inspecting {label} ({i + 1}/{len(uploaded_files)})…",
        )

        dest_name = f"{i}_{uploaded_file.name}"
        image_bytes = uploaded_file.getvalue()

        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        quality = assess_quality(decoded)

        try:
            result = run_inspection_cached(
                pipeline, image_bytes, dest_name,
                confidence, iou,
                mode_settings["imgsz"], mode_settings["augment"], mode_settings["tiled"],
            )
        except Exception as e:
            st.error(f"Failed to process **{label}**: {e}")
            continue

        result = dict(result)
        result["image"] = uploaded_file.name  # keep the human-friendly name for reports
        results.append((label, SESSION_DIR / dest_name, result, quality))

    progress.progress(1.0, text="Done.")
    progress.empty()

    if not results:
        return

    st.divider()

    if len(results) == 1:
        label, image_path, result, quality = results[0]
        render_single_result(image_path, result, quality, key_suffix="0")
    else:
        tabs = st.tabs([label for label, _, _, _ in results])
        for tab, (i, (label, image_path, result, quality)) in zip(tabs, enumerate(results)):
            with tab:
                render_single_result(image_path, result, quality, key_suffix=str(i))

    st.markdown(
        "<div class='pcb-footer'>AI-assisted inspection — verify Critical and High severity "
        "findings manually before final disposition.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
