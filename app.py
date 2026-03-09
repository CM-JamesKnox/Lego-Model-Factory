"""
app.py — LEGO Synthetic Data & YOLO Training Factory
Streamlit frontend: Generate → Inspect → Train → Test
"""

import os
import re
import glob
import shutil
import random
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Config — adjust these paths if needed
# ---------------------------------------------------------------------------

BLENDER_EXE  = r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
LDRAW_ROOT   = r"C:\Program Files\Studio 2.0\ldraw"
LDRAW_PARTS  = os.path.join(LDRAW_ROOT, "parts")
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASETS_RAW = PROJECT_ROOT / "datasets" / "raw"
DATASETS_YOLO= PROJECT_ROOT / "datasets" / "yolo"
MODELS_DIR   = PROJECT_ROOT / "models"
HDRI_DIR     = PROJECT_ROOT / "hdri"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

for d in (DATASETS_RAW, DATASETS_YOLO, MODELS_DIR, HDRI_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Scanning LDraw library…")
def load_parts_map() -> dict[str, str]:
    """Scan the LDraw parts directory and return {part_id: description}."""
    parts_map = {}
    if not os.path.isdir(LDRAW_PARTS):
        return parts_map
    for fname in sorted(os.listdir(LDRAW_PARTS)):
        if not fname.lower().endswith(".dat"):
            continue
        part_id = fname[:-4]  # strip .dat
        fpath   = os.path.join(LDRAW_PARTS, fname)
        desc    = part_id
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith("0 ") and not any(
                        kw in stripped for kw in
                        ("Name:", "Author:", "!LDRAW_ORG", "!LICENSE", "BFC", "!HISTORY",
                         "!KEYWORDS", "!CATEGORY", "!CMDLINE")
                    ):
                        candidate = stripped[2:].strip()
                        if candidate:
                            desc = candidate
                    break  # only need first meaningful comment line
        except OSError:
            pass
        parts_map[part_id] = desc
    return parts_map


def draw_yolo_boxes(img_bgr: np.ndarray, label_path: str,
                    color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Overlay YOLO bounding boxes onto an OpenCV image."""
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as lf:
        for line in lf:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), *[float(p) for p in parts[1:]]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def stream_subprocess(cmd: list[str], progress_parts: list[str]):
    """
    Run a subprocess in a background thread.
    Writes only to st.session_state — never calls Streamlit widget methods
    directly (those must run on the main script thread).
    """
    progress_re = re.compile(r"PROGRESS:([^:]+):(\d+)/(\d+)")
    part_progress = {p: 0 for p in progress_parts}
    total = st.session_state.get("gen_count", 100) * len(progress_parts)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    st.session_state["gen_proc"] = proc

    log_lines: list[str] = []
    for raw in proc.stdout:
        line = raw.rstrip()
        m = progress_re.match(line)
        if m:
            part_id, done = m.group(1), int(m.group(2))
            part_progress[part_id] = done
            total_done = sum(part_progress.values())
            st.session_state["gen_progress"] = total_done / max(total, 1)
        else:
            log_lines.append(line)
            if len(log_lines) > 80:
                log_lines = log_lines[-80:]
        # Write the full log buffer to session_state so the main thread can render it
        st.session_state["gen_log"] = "\n".join(log_lines)

    proc.wait()
    st.session_state["gen_running"] = False
    st.session_state["gen_proc"]    = None
    st.session_state["gen_status"]  = "done" if proc.returncode == 0 \
                                       else f"error (code {proc.returncode})"


def prepare_yolo_dataset(raw_root: Path, yolo_root: Path, val_split: float = 0.2):
    """
    Flatten datasets/raw/{part_id}/images/*.png → datasets/yolo/images/{train,val}/
    and matching labels.  Returns list of class names in order.
    """
    # Discover parts (class names = sorted part IDs)
    part_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    class_names = [d.name for d in part_dirs]

    # Build flat list of (img_path, lbl_path, class_id)
    all_samples = []
    for class_id, part_dir in enumerate(part_dirs):
        img_dir = part_dir / "images"
        lbl_dir = part_dir / "labels"
        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                all_samples.append((img_path, lbl_path, class_id))

    random.shuffle(all_samples)
    split_idx  = int(len(all_samples) * (1 - val_split))
    train_set  = all_samples[:split_idx]
    val_set    = all_samples[split_idx:]

    for subset_name, subset in [("train", train_set), ("val", val_set)]:
        img_out = yolo_root / "images" / subset_name
        lbl_out = yolo_root / "labels" / subset_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path, _ in subset:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    return class_names


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LEGO YOLO Factory",
    page_icon="🧱",
    layout="wide",
)

st.title("LEGO Synthetic Data & YOLO Training Factory")
tab_gen, tab_inspect, tab_train, tab_test = st.tabs(
    ["Generate", "Inspect", "Train", "Test"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Generate
# ---------------------------------------------------------------------------

with tab_gen:
    st.header("Synthetic Data Generation")

    parts_map = load_parts_map()

    if not parts_map:
        st.error(f"No .dat files found in `{LDRAW_PARTS}`. "
                 "Check that Studio 2.0 is installed and LDRAW_PARTS is correct.")
        st.stop()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Search box to filter the multi-select list
        search_term = st.text_input("Filter parts (ID or description)",
                                    placeholder="e.g. brick, 3001, plate…")
        filtered = {
            pid: desc for pid, desc in parts_map.items()
            if search_term.lower() in pid.lower()
            or search_term.lower() in desc.lower()
        } if search_term else parts_map

        selected_parts = st.multiselect(
            f"Select parts ({len(filtered):,} shown)",
            options=list(filtered.keys()),
            format_func=lambda x: f"{x}  —  {parts_map.get(x, '')}",
            placeholder="Type to search, then select…",
        )

    with col_right:
        gen_count = st.slider("Images per part", 50, 500, 100, 10)
        resolution = st.selectbox("Resolution", ["640x640", "1280x1280", "320x320"])
        cycles_samples = st.slider("Cycles samples (quality vs speed)", 32, 256, 96, 16)
        out_dir = st.text_input("Output directory", str(DATASETS_RAW))
        hdri_path = st.text_input("HDRI directory (optional)", str(HDRI_DIR))

    st.divider()

    # State init
    for key, val in [
        ("gen_running", False), ("gen_progress", 0.0),
        ("gen_status", ""), ("gen_proc", None), ("gen_count", gen_count),
        ("gen_log", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.session_state["gen_count"] = gen_count

    col_btn, col_stop = st.columns([1, 1])

    with col_btn:
        start = st.button("Generate", type="primary",
                          disabled=st.session_state["gen_running"] or not selected_parts)

    with col_stop:
        if st.button("Stop", disabled=not st.session_state["gen_running"]):
            proc = st.session_state.get("gen_proc")
            if proc:
                proc.terminate()
            st.session_state["gen_running"] = False

    progress_bar = st.progress(st.session_state["gen_progress"])
    status_text  = st.empty()
    log_box      = st.empty()

    if start and selected_parts:
        st.session_state["gen_running"] = True
        st.session_state["gen_progress"] = 0.0
        st.session_state["gen_status"]   = "running"

        cmd = [
            BLENDER_EXE, "--background", "--python",
            str(PROJECT_ROOT / "blender_gen.py"), "--",
            "--parts",      ",".join(selected_parts),
            "--count",      str(gen_count),
            "--output",     out_dir,
            "--ldraw",      LDRAW_ROOT,
            "--resolution", resolution,
            "--samples",    str(cycles_samples),
        ]
        if os.path.isdir(hdri_path):
            cmd += ["--hdri_dir", hdri_path]

        st.session_state["gen_log"] = ""
        t = threading.Thread(
            target=stream_subprocess,
            args=(cmd, selected_parts),
            daemon=True,
        )
        t.start()
        st.rerun()

    # Main thread renders whatever the background thread wrote to session_state
    if st.session_state["gen_running"]:
        progress_bar.progress(st.session_state["gen_progress"])
        status_text.info(f"Rendering… {st.session_state['gen_progress']:.1%} complete")
        if st.session_state["gen_log"]:
            log_box.code(st.session_state["gen_log"])
        time.sleep(0.5)
        st.rerun()
    elif st.session_state["gen_status"] == "done":
        progress_bar.progress(1.0)
        status_text.success("Generation complete!")
        if st.session_state["gen_log"]:
            log_box.code(st.session_state["gen_log"])
    elif st.session_state["gen_status"].startswith("error"):
        status_text.error(f"Blender exited with: {st.session_state['gen_status']}")
        if st.session_state["gen_log"]:
            log_box.code(st.session_state["gen_log"])


# ---------------------------------------------------------------------------
# TAB 2 — Inspect
# ---------------------------------------------------------------------------

with tab_inspect:
    st.header("Dataset Inspector")

    raw_root = Path(str(DATASETS_RAW))
    part_dirs = sorted([d.name for d in raw_root.iterdir() if d.is_dir()]) \
        if raw_root.exists() else []

    if not part_dirs:
        st.info("No generated data yet. Run the Generate tab first.")
    else:
        col_a, col_b, col_c = st.columns([1, 1, 1])

        with col_a:
            filter_part = st.selectbox("Filter by part", ["All"] + part_dirs)
        with col_b:
            grid_cols = st.slider("Gallery columns", 2, 6, 3)
        with col_c:
            max_images = st.slider("Max images shown", 10, 200, 50, 10)

        # Collect stats
        stats = {}
        for pd in part_dirs:
            img_dir = raw_root / pd / "images"
            stats[pd] = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0

        st.subheader("Class summary")
        stat_cols = st.columns(min(len(stats), 6))
        for i, (pid, cnt) in enumerate(stats.items()):
            stat_cols[i % 6].metric(pid, f"{cnt} imgs")

        st.divider()

        # Gather image paths
        search_parts = [filter_part] if filter_part != "All" else part_dirs
        img_paths = []
        for sp in search_parts:
            img_dir = raw_root / sp / "images"
            if img_dir.exists():
                img_paths += sorted(img_dir.glob("*.png"))

        img_paths = img_paths[:max_images]

        if not img_paths:
            st.warning("No images found for selected filter.")
        else:
            st.write(f"Showing {len(img_paths)} images")
            rows = [img_paths[i:i+grid_cols] for i in range(0, len(img_paths), grid_cols)]
            for row in rows:
                cols = st.columns(grid_cols)
                for col, img_path in zip(cols, row):
                    part_id  = img_path.parent.parent.name
                    lbl_path = str(img_path).replace(
                        os.sep + "images" + os.sep,
                        os.sep + "labels" + os.sep
                    ).replace(".png", ".txt")

                    img_bgr  = cv2.imread(str(img_path))
                    if img_bgr is None:
                        col.warning(f"Cannot load {img_path.name}")
                        continue

                    annotated = draw_yolo_boxes(img_bgr, lbl_path)
                    col.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"{part_id} / {img_path.name}",
                        use_container_width=True,
                    )


# ---------------------------------------------------------------------------
# TAB 3 — Train
# ---------------------------------------------------------------------------

with tab_train:
    st.header("YOLO11 Training")

    col1, col2 = st.columns([1, 1])

    with col1:
        model_name  = st.text_input("Model name", "lego_yolo11")
        base_model  = st.selectbox("Base model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
        epochs      = st.slider("Epochs", 10, 300, 50)
        imgsz       = st.selectbox("Image size", [640, 320, 1280])
        val_split   = st.slider("Validation split", 0.05, 0.40, 0.20, 0.05)
        batch_size  = st.number_input("Batch size (-1 = auto)", value=-1, step=1)

    with col2:
        st.subheader("Dataset status")
        raw_root_t = DATASETS_RAW
        part_dirs_t = sorted([d.name for d in raw_root_t.iterdir() if d.is_dir()]) \
            if raw_root_t.exists() else []
        if part_dirs_t:
            total_imgs = sum(
                len(list((raw_root_t / p / "images").glob("*.png")))
                for p in part_dirs_t
                if (raw_root_t / p / "images").exists()
            )
            st.metric("Classes",      len(part_dirs_t))
            st.metric("Total images", total_imgs)
            st.write("**Classes:**", ", ".join(part_dirs_t))
        else:
            st.info("No data in datasets/raw yet.")

    st.divider()

    prep_col, train_col = st.columns([1, 1])

    with prep_col:
        if st.button("Prepare Dataset (train/val split)"):
            if not part_dirs_t:
                st.error("No raw data found.")
            else:
                with st.spinner("Splitting dataset…"):
                    # Wipe and rebuild yolo dir
                    if DATASETS_YOLO.exists():
                        shutil.rmtree(DATASETS_YOLO)
                    class_names = prepare_yolo_dataset(
                        DATASETS_RAW, DATASETS_YOLO, val_split=val_split
                    )

                    yaml_data = {
                        "path":  str(DATASETS_YOLO.resolve()),
                        "train": "images/train",
                        "val":   "images/val",
                        "nc":    len(class_names),
                        "names": class_names,
                    }
                    with open(DATASET_YAML, "w") as yf:
                        yaml.dump(yaml_data, yf, default_flow_style=False)

                st.success(
                    f"Dataset prepared: {len(class_names)} classes. "
                    f"YAML written to {DATASET_YAML}"
                )

    with train_col:
        if "train_running" not in st.session_state:
            st.session_state["train_running"] = False

        start_train = st.button(
            "Start Training",
            type="primary",
            disabled=st.session_state["train_running"] or not DATASET_YAML.exists(),
        )

    train_log = st.empty()

    if start_train:
        st.session_state["train_running"] = True
        cmd = [
            "yolo", "train",
            f"model={base_model}",
            f"data={DATASET_YAML}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch_size}",
            f"project={MODELS_DIR}",
            f"name={model_name}",
        ]
        log_lines = []

        def _run_training():
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for raw in proc.stdout:
                log_lines.append(raw.rstrip())
                if len(log_lines) > 100:
                    log_lines.pop(0)
                train_log.code("\n".join(log_lines))
            proc.wait()
            st.session_state["train_running"] = False
            if proc.returncode == 0:
                st.session_state["train_status"] = "done"
            else:
                st.session_state["train_status"] = f"error {proc.returncode}"

        threading.Thread(target=_run_training, daemon=True).start()
        st.rerun()

    if st.session_state.get("train_status") == "done":
        best = list(MODELS_DIR.glob(f"**/{model_name}/weights/best.pt"))
        if best:
            st.success(f"Training complete! Best model: {best[0]}")
        else:
            st.success("Training complete!")


# ---------------------------------------------------------------------------
# TAB 4 — Test (Webcam)
# ---------------------------------------------------------------------------

with tab_test:
    st.header("Live Webcam Inference")

    pt_files = sorted(MODELS_DIR.glob("**/weights/best.pt")) + \
               sorted(MODELS_DIR.glob("**/weights/last.pt"))
    pt_labels = [str(p.relative_to(PROJECT_ROOT)) for p in pt_files]

    if not pt_labels:
        st.info("No trained models found. Train a model in the Train tab first.")
    else:
        col_t1, col_t2 = st.columns([1, 2])

        with col_t1:
            selected_model = st.selectbox("Select model", pt_labels)
            conf_thresh    = st.slider("Confidence threshold", 0.10, 1.00, 0.50, 0.05)
            iou_thresh     = st.slider("IoU threshold (NMS)", 0.10, 1.00, 0.45, 0.05)
            cam_index      = st.number_input("Camera index", 0, 10, 0)

            start_cam = st.button("Start Webcam", type="primary",
                                  disabled=st.session_state.get("cam_running", False))
            stop_cam  = st.button("Stop Webcam",
                                  disabled=not st.session_state.get("cam_running", False))

        with col_t2:
            frame_slot = st.empty()
            fps_slot   = st.empty()

        # Session state
        for k, v in [("cam_running", False), ("cam_stop", False)]:
            if k not in st.session_state:
                st.session_state[k] = v

        if stop_cam:
            st.session_state["cam_stop"]    = True
            st.session_state["cam_running"] = False

        if start_cam and selected_model:
            from ultralytics import YOLO as _YOLO

            st.session_state["cam_running"] = True
            st.session_state["cam_stop"]    = False

            model_path = PROJECT_ROOT / selected_model
            model      = _YOLO(str(model_path))
            cap        = cv2.VideoCapture(int(cam_index))

            if not cap.isOpened():
                st.error(f"Cannot open camera {cam_index}.")
                st.session_state["cam_running"] = False
            else:
                prev_time = time.time()
                while cap.isOpened() and not st.session_state.get("cam_stop", False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results    = model(frame, conf=conf_thresh, iou=iou_thresh,
                                       verbose=False)
                    annotated  = results[0].plot()
                    rgb_frame  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    now = time.time()
                    fps = 1.0 / max(now - prev_time, 1e-6)
                    prev_time = now

                    frame_slot.image(rgb_frame, channels="RGB",
                                     use_container_width=True)
                    fps_slot.metric("FPS", f"{fps:.1f}")

                cap.release()
                st.session_state["cam_running"] = False
                st.session_state["cam_stop"]    = False
