"""
=========================================================
PCB Defect Inspection Tool - Streamlit UI

Purpose
-------
Upload one or more PCB images, run each through the full
inspection pipeline (YOLO detection + Gemini explanation),
and display an annotated image plus a defect-by-defect
report in plain language, with downloadable results.

Run with:
    streamlit run app.py

Author : Manjeet
=========================================================
"""

import json
import os
import tempfile
import uuid
from pathlib import Path

import cv2
import streamlit as st

from services.pipeline import PCBInspectionPipeline

# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔍",
    layout="wide",
)

SEVERITY_COLORS = {
    "Critical": "#ff4b4b",
    "High": "#ff8c42",
    "Medium": "#f4c542",
    "Low": "#4caf50",
    "Unknown": "#999999",
}

SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Unknown": 4}

BOX_COLOR_BGR = {
    "Critical": (0, 0, 255),
    "High": (0, 140, 255),
    "Medium": (0, 210, 255),
    "Low": (0, 200, 0),
    "Unknown": (180, 180, 180),
}

# Per-session scratch dir so concurrent users / repeat uploads
# never collide on filenames, and everything is easy to clean up.
SESSION_DIR = Path(tempfile.gettempdir()) / "pcb_ai_sessions" / str(
    st.session_state.get("_session_id") or uuid.uuid4()
)
st.session_state["_session_id"] = SESSION_DIR.name
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Startup checks — fail loudly and helpfully, not mid-run
# =========================================================

def check_environment() -> list[str]:
    problems = []

    if not os.getenv("GEMINI_API_KEY"):
        problems.append(
            "`GEMINI_API_KEY` is not set. Add it to a `.env` file in the "
            "project root (`GEMINI_API_KEY=your_key_here`) or set it as an "
            "environment variable, then restart the app."
        )

    model_path = (
        Path(__file__).resolve().parent
        / "runs" / "pcb_detector-3" / "weights" / "best.pt"
    )
    if not model_path.exists():
        problems.append(
            f"Trained model not found at `{model_path}`. Update the path in "
            "`training/inference.py` if your run folder has a different name."
        )

    return problems


# =========================================================
# Cache the pipeline so the model loads only once,
# not on every rerun / upload.
# =========================================================

@st.cache_resource(show_spinner="Loading detection model...")
def load_pipeline():
    return PCBInspectionPipeline()


def sorted_by_severity(detections: list) -> list:
    return sorted(
        detections,
        key=lambda d: SEVERITY_RANK.get(
            d.get("explanation", {}).get("severity", "Unknown"), 4
        ),
    )


def overall_verdict(detections: list) -> tuple[str, str, str]:
    """Returns (label, color, message) for a pass/fail-style banner."""
    if not detections:
        return "PASS", "#4caf50", "No defects detected on this board."

    severities = {d.get("explanation", {}).get("severity", "Unknown") for d in detections}

    if "Critical" in severities:
        return "FAIL", "#ff4b4b", "Critical defect(s) found — board should not ship as-is."
    if "High" in severities:
        return "REVIEW", "#ff8c42", "High-severity defect(s) found — needs inspection before release."
    if "Medium" in severities:
        return "REVIEW", "#f4c542", "Medium-severity defect(s) found — recommend rework."
    return "MINOR", "#4caf50", "Only low-severity defects found."


def draw_boxes(image_path: Path, detections: list):
    """Draw bounding boxes + labels, colored by severity, on the image."""
    img = cv2.imread(str(image_path))

    for det in detections:
        bbox = det["bbox"]
        severity = det.get("explanation", {}).get("severity", "Unknown")
        color = BOX_COLOR_BGR.get(severity, (180, 180, 180))

        x1, y1, x2, y2 = (
            int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{det.get('id', '?')} {det['class']}"
        cv2.putText(
            img, label, (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_thumbnail(image_path: Path, bbox: dict, padding: int = 10):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x1 = max(0, int(bbox["x1"]) - padding)
    y1 = max(0, int(bbox["y1"]) - padding)
    x2 = min(w, int(bbox["x2"]) + padding)
    y2 = min(h, int(bbox["y2"]) + padding)
    crop = img[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def build_markdown_report(result: dict) -> str:
    lines = [f"# PCB Inspection Report — {result['image']}", ""]
    label, _, message = overall_verdict(result["detections"])
    lines.append(f"**Verdict:** {label} — {message}")
    lines.append(f"**Total defects:** {result['total_defects']}")
    lines.append(f"**Processed in:** {result.get('pipeline_time_sec', '?')}s")
    lines.append("")

    for det in sorted_by_severity(result["detections"]):
        exp = det.get("explanation", {})
        lines.append(f"## Defect #{det.get('id', '?')} — {det['class'].replace('_', ' ').title()}")
        lines.append(f"- **Severity:** {exp.get('severity', 'N/A')}")
        lines.append(f"- **Confidence:** {det['confidence']:.1%}")
        lines.append(f"- **What's wrong:** {exp.get('explanation', 'N/A')}")
        lines.append(f"- **Likely cause:** {exp.get('root_cause', 'N/A')}")
        lines.append(f"- **Recommended fix:** {exp.get('recommended_fix', 'N/A')}")
        lines.append(f"- **Prevention:** {exp.get('prevention', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def render_defect_card(det: dict, image_path: Path):
    exp = det.get("explanation", {})
    severity = exp.get("severity", "Unknown")
    color = SEVERITY_COLORS.get(severity, "#999999")

    with st.container(border=True):
        thumb_col, info_col = st.columns([1, 3])

        with thumb_col:
            try:
                st.image(crop_thumbnail(image_path, det["bbox"]), use_container_width=True)
            except Exception:
                pass

        with info_col:
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"### #{det.get('id', '?')} — {det['class'].replace('_', ' ').title()}")
            with top_right:
                st.markdown(
                    f"<span style='background-color:{color};color:white;"
                    f"padding:4px 10px;border-radius:6px;font-weight:600;'>"
                    f"{severity}</span>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Confidence: {det['confidence']:.1%}")
            st.markdown(f"**What's wrong:** {exp.get('explanation', 'N/A')}")
            st.markdown(f"**Likely cause:** {exp.get('root_cause', 'N/A')}")
            st.markdown(f"**Recommended fix:** {exp.get('recommended_fix', 'N/A')}")
            st.markdown(f"**Prevention:** {exp.get('prevention', 'N/A')}")


def render_single_result(image_path: Path, result: dict):
    detections = sorted_by_severity(result.get("detections", []))
    total = result.get("total_defects", 0)
    elapsed = result.get("pipeline_time_sec", None)

    verdict_label, verdict_color, verdict_msg = overall_verdict(detections)
    st.markdown(
        f"<div style='background-color:{verdict_color};color:white;padding:12px 16px;"
        f"border-radius:8px;font-size:1.1em;font-weight:600;margin-bottom:12px;'>"
        f"{verdict_label} — {verdict_msg}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Annotated Image")
        if detections:
            st.image(draw_boxes(image_path, detections), use_container_width=True)
        else:
            st.image(str(image_path), use_container_width=True)

    with right:
        st.subheader("Summary")
        st.metric("Defects Found", total)
        if elapsed is not None:
            st.caption(f"Processed in {elapsed}s")

        if total:
            severities = [d.get("explanation", {}).get("severity", "Unknown") for d in detections]
            for sev in ["Critical", "High", "Medium", "Low"]:
                count = severities.count(sev)
                if count:
                    st.write(f"**{sev}:** {count}")

        st.download_button(
            "⬇ Download JSON report",
            data=json.dumps(result, indent=2),
            file_name=f"{Path(result['image']).stem}_report.json",
            mime="application/json",
            key=f"json_{result['image']}",
        )
        st.download_button(
            "⬇ Download Markdown report",
            data=build_markdown_report(result),
            file_name=f"{Path(result['image']).stem}_report.md",
            mime="text/markdown",
            key=f"md_{result['image']}",
        )

    if detections:
        st.divider()
        st.subheader("Defect Report")
        st.caption("Sorted by severity, most critical first.")
        for det in detections:
            render_defect_card(det, image_path)


# =========================================================
# Main app
# =========================================================

def main():
    st.title("🔍 PCB Defect Inspection Tool")
    st.write(
        "Upload one or more PCB images to automatically detect defects and "
        "get plain-language explanations, root causes, and fixes."
    )

    problems = check_environment()
    if problems:
        st.error("Setup incomplete:\n\n" + "\n\n".join(f"- {p}" for p in problems))
        return

    with st.sidebar:
        st.header("Detection settings")
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
        iou = st.slider("IoU threshold", 0.05, 0.95, 0.40, 0.05)
        st.caption(
            "Lower confidence = catches more possible defects, but more "
            "false positives. Adjust and re-run if results look off."
        )

    uploaded_files = st.file_uploader(
        "Upload PCB image(s)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more images to begin inspection.")
        return

    pipeline = load_pipeline()

    results_by_name = {}
    image_paths = {}

    progress = st.progress(0.0, text="Starting...")
    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(
            i / len(uploaded_files),
            text=f"Inspecting {uploaded_file.name} ({i + 1}/{len(uploaded_files)})...",
        )

        dest = SESSION_DIR / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())

        try:
            result = pipeline.inspect(dest, confidence=confidence, iou=iou)
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")
            continue

        results_by_name[uploaded_file.name] = result
        image_paths[uploaded_file.name] = dest

    progress.progress(1.0, text="Done.")
    progress.empty()

    if not results_by_name:
        return

    st.divider()

    if len(results_by_name) == 1:
        name = next(iter(results_by_name))
        render_single_result(image_paths[name], results_by_name[name])
    else:
        tabs = st.tabs(list(results_by_name.keys()))
        for tab, name in zip(tabs, results_by_name.keys()):
            with tab:
                render_single_result(image_paths[name], results_by_name[name])


if __name__ == "__main__":
    main()
