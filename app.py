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

import base64
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import yaml
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Config — adjust these paths if needed
# ---------------------------------------------------------------------------

BLENDER_EXE  = r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
LDRAW_ROOT   = r"C:\Program Files\Studio 2.0\ldraw"
LDRAW_PARTS  = os.path.join(LDRAW_ROOT, "parts")
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASETS_RAW  = PROJECT_ROOT / "datasets" / "raw"
DATASETS_YOLO = PROJECT_ROOT / "datasets" / "yolo"
REAL_IMAGES_DIR = PROJECT_ROOT / "datasets" / "real_images"
MODELS_DIR    = PROJECT_ROOT / "models"
HDRI_DIR      = PROJECT_ROOT / "hdri"
DATASET_YAML  = PROJECT_ROOT / "dataset.yaml"

for d in (DATASETS_RAW, DATASETS_YOLO, REAL_IMAGES_DIR, MODELS_DIR, HDRI_DIR):
    d.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def _start_bbox_server():
    """Serve bbox_component over a plain HTTP server to avoid path-with-spaces issues."""
    import socket
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    _comp_dir = str(PROJECT_ROOT / "bbox_component")
    with socket.socket() as _s:
        _s.bind(("", 0))
        _port = _s.getsockname()[1]
    class _H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=_comp_dir, **kw)
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            super().end_headers()
        def log_message(self, *a, **kw): pass
    threading.Thread(
        target=HTTPServer(("localhost", _port), _H).serve_forever,
        daemon=True,
    ).start()
    return _port

_bbox_editor = components.declare_component(
    "bbox_editor", url=f"http://localhost:{_start_bbox_server()}"
)

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


def prepare_yolo_dataset(raw_root: Path, yolo_root: Path, val_split: float = 0.2,
                         real_root: Path | None = None):
    """
    Flatten datasets/raw/{part_id}/images/*.png → datasets/yolo/images/{train,val}/
    and matching labels. Optionally merges real-world images from real_root.
    Returns list of class names in order.
    """
    # Discover synthetic parts
    part_dirs   = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    class_names = [d.name for d in part_dirs]

    # Add any real-world-only parts that aren't in synthetic data
    if real_root and real_root.exists():
        for d in sorted(real_root.iterdir()):
            if d.is_dir() and d.name not in class_names:
                class_names.append(d.name)

    class_id_map = {name: i for i, name in enumerate(class_names)}

    # Synthetic samples — (img_path, lbl_path, class_id, is_real)
    all_samples = []
    for part_dir in part_dirs:
        class_id = class_id_map[part_dir.name]
        img_dir  = part_dir / "images"
        lbl_dir  = part_dir / "labels"
        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                all_samples.append((img_path, lbl_path, class_id, False))

    # Real-world samples (labels saved with class_id=0 per-part, need remapping)
    if real_root and real_root.exists():
        for real_part_dir in sorted(real_root.iterdir()):
            if not real_part_dir.is_dir():
                continue
            class_id = class_id_map.get(real_part_dir.name)
            if class_id is None:
                continue
            img_dir = real_part_dir / "images"
            lbl_dir = real_part_dir / "labels"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    all_samples.append((img_path, lbl_path, class_id, True))

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_split))
    train_set = all_samples[:split_idx]
    val_set   = all_samples[split_idx:]

    for subset_name, subset in [("train", train_set), ("val", val_set)]:
        img_out = yolo_root / "images" / subset_name
        lbl_out = yolo_root / "labels" / subset_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path, class_id, is_real in subset:
            # Avoid filename collisions between synthetic and real images
            dest_img = img_out / img_path.name
            if dest_img.exists():
                dest_img = img_out / f"r_{img_path.name}"
            shutil.copy2(img_path, dest_img)

            if is_real:
                # Remap placeholder class_id 0 → actual class_id for this part
                with open(lbl_path) as f:
                    lines = f.readlines()
                remapped = []
                for line in lines:
                    parts_l = line.strip().split()
                    if len(parts_l) == 5:
                        parts_l[0] = str(class_id)
                        remapped.append(" ".join(parts_l))
                with open(lbl_out / (dest_img.stem + ".txt"), "w") as f:
                    f.write("\n".join(remapped))
            else:
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
tab_gen, tab_inspect, tab_train, tab_test, tab_boost = st.tabs(
    ["Generate", "Inspect", "Train", "Test", "Real-World Booster"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Generate
# ---------------------------------------------------------------------------

with tab_gen:
    st.header("Synthetic Data Generation")

    parts_map = load_parts_map()

    if not parts_map:
        st.warning(f"No .dat files found in `{LDRAW_PARTS}`. "
                   "Check that Studio 2.0 is installed and LDRAW_PARTS is correct. "
                   "The Generate tab is disabled, but other tabs still work.")
    else:
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
            gpu_device = st.selectbox(
                "GPU backend",
                ["AUTO", "OPTIX", "CUDA", "HIP", "METAL", "CPU"],
                help="AUTO tries OptiX → CUDA → HIP → Metal then falls back to CPU. "
                     "Pick OPTIX for NVIDIA RTX cards, CUDA for older NVIDIA, HIP for AMD.",
            )
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
                "--parts",       ",".join(selected_parts),
                "--count",       str(gen_count),
                "--output",      out_dir,
                "--ldraw",       LDRAW_ROOT,
                "--resolution",  resolution,
                "--samples",     str(cycles_samples),
                "--device_type", gpu_device,
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
                        DATASETS_RAW, DATASETS_YOLO, val_split=val_split,
                        real_root=REAL_IMAGES_DIR,
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
        st.session_state["train_running"]   = True
        st.session_state["train_log_lines"] = []
        st.session_state["train_status"]    = "running"
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

        def _run_training():
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for raw in proc.stdout:
                # Thread only writes to session state — no Streamlit UI calls
                lines = st.session_state["train_log_lines"]
                lines.append(raw.rstrip())
                if len(lines) > 200:
                    lines.pop(0)
            proc.wait()
            st.session_state["train_running"] = False
            st.session_state["train_status"] = (
                "done" if proc.returncode == 0 else f"error {proc.returncode}"
            )

        threading.Thread(target=_run_training, daemon=True).start()
        st.rerun()

    # Display live log — main thread polls session state while training runs
    if st.session_state.get("train_running") or st.session_state.get("train_log_lines"):
        _lines = st.session_state.get("train_log_lines", [])
        train_log.code("\n".join(_lines) if _lines else "Starting…")
        if st.session_state.get("train_running"):
            time.sleep(1)
            st.rerun()

    if st.session_state.get("train_status") == "done":
        best = list(MODELS_DIR.glob(f"**/{model_name}/weights/best.pt"))
        if best:
            st.success(f"Training complete! Best model: {best[0]}")
        else:
            st.success("Training complete!")


# ---------------------------------------------------------------------------
# TAB 4 — Test (Webcam / Upload)
# ---------------------------------------------------------------------------

with tab_test:
    st.header("YOLO Inference")

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

            input_mode = st.radio("Input source", ["Webcam", "Upload Image"],
                                  horizontal=True)

            if input_mode == "Webcam":
                cam_index = st.number_input("Camera index", 0, 10, 0)
                start_cam = st.button("Start Webcam", type="primary",
                                      disabled=st.session_state.get("cam_running", False))
                stop_cam  = st.button("Stop Webcam",
                                      disabled=not st.session_state.get("cam_running", False))
            else:
                uploaded_file = st.file_uploader(
                    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
                )
                run_upload = st.button("Run Inference", type="primary",
                                       disabled=uploaded_file is None)

        with col_t2:
            frame_slot = st.empty()
            fps_slot   = st.empty()

        # Session state
        for k, v in [("cam_running", False), ("cam_stop", False)]:
            if k not in st.session_state:
                st.session_state[k] = v

        # --- Webcam mode ---
        if input_mode == "Webcam":
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

                        results   = model(frame, conf=conf_thresh, iou=iou_thresh,
                                          verbose=False)
                        annotated = results[0].plot()
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        now = time.time()
                        fps = 1.0 / max(now - prev_time, 1e-6)
                        prev_time = now

                        frame_slot.image(rgb_frame, channels="RGB",
                                         use_container_width=True)
                        fps_slot.metric("FPS", f"{fps:.1f}")

                    cap.release()
                    st.session_state["cam_running"] = False
                    st.session_state["cam_stop"]    = False

        # --- Upload Image mode ---
        else:
            if run_upload and uploaded_file is not None:
                from ultralytics import YOLO as _YOLO

                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                model_path = PROJECT_ROOT / selected_model
                model      = _YOLO(str(model_path))

                results   = model(img_bgr, conf=conf_thresh, iou=iou_thresh,
                                  verbose=False)
                annotated = results[0].plot()
                rgb_out   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                frame_slot.image(rgb_out, channels="RGB", use_container_width=True)

                # Show detection summary
                boxes = results[0].boxes
                if boxes is not None and len(boxes):
                    names = results[0].names
                    detections = [
                        f"{names[int(cls)]} ({conf:.0%})"
                        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist())
                    ]
                    fps_slot.success(f"Detected: {', '.join(detections)}")
                else:
                    fps_slot.info("No detections above threshold.")


# ---------------------------------------------------------------------------
# TAB 5 — Real-World Booster
# ---------------------------------------------------------------------------

with tab_boost:
    st.header("Real-World Booster")
    st.caption(
        "Download real LEGO images from Bing and draw bounding boxes to create "
        "extra training data. Run this while synthetic images are generating — "
        "they will be automatically included the next time you Prepare Dataset."
    )

    # --- Controls ---
    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        boost_part_id = st.text_input(
            "LEGO Part ID", placeholder="e.g. 3001", key="boost_part_input"
        )
    with b_col2:
        imgs_per_query = st.number_input(
            "Images per search query", min_value=10, max_value=50, value=25, step=5,
            help="Two queries run per part (topside + underside), so total ≈ 2×this value."
        )

    # Session state defaults
    for _k, _v in [
        ("boost_images",        []),
        ("boost_idx",           0),
        ("boost_downloading",   False),
        ("boost_download_done", False),
        ("boost_current_part",  ""),
        ("boost_labeled",       0),
        ("boost_skipped",       0),
        ("boost_download_error", ""),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Auto-restore after page refresh: if session was cleared but downloads exist on disk,
    # reload them automatically so the user doesn't need to re-download.
    if not st.session_state["boost_images"] and not st.session_state["boost_downloading"] \
            and boost_part_id:
        _auto_root = REAL_IMAGES_DIR / boost_part_id / "raw_downloads"
        if _auto_root.exists():
            _auto_imgs = []
            for _ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"):
                _auto_imgs += sorted(_auto_root.rglob(_ext))
            if _auto_imgs:
                st.session_state["boost_images"]       = [str(p) for p in _auto_imgs]
                st.session_state["boost_current_part"] = boost_part_id
                st.session_state["boost_download_done"] = True

    fetch_btn = st.button(
        "Fetch Images from Google",
        type="primary",
        disabled=not boost_part_id or st.session_state["boost_downloading"],
    )

    if fetch_btn and boost_part_id:
        st.session_state["boost_images"]        = []
        st.session_state["boost_idx"]           = 0
        st.session_state["boost_downloading"]   = True
        st.session_state["boost_download_done"] = False
        st.session_state["boost_current_part"]  = boost_part_id
        st.session_state["boost_labeled"]       = 0
        st.session_state["boost_skipped"]       = 0
        st.session_state["boost_download_error"] = ""

        dl_root = REAL_IMAGES_DIR / boost_part_id / "raw_downloads"
        if dl_root.exists():
            shutil.rmtree(dl_root)
        dl_root.mkdir(parents=True, exist_ok=True)

        def _download_images(part_id, n_per_query, root):
            import logging
            # Suppress all icrawler loggers (includes child loggers like downloader/parser)
            for _log_name in logging.Logger.manager.loggerDict:
                if "icrawler" in _log_name or "crawler" in _log_name.lower():
                    logging.getLogger(_log_name).setLevel(logging.CRITICAL)
            logging.getLogger("icrawler").setLevel(logging.CRITICAL)

            try:
                from icrawler.builtin import BingImageCrawler
                queries = {
                    "topside":   f"Lego brick {part_id}",
                    "underside": f"Lego brick {part_id} underside",
                }
                for label, query in queries.items():
                    out_dir = root / label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
                    crawler.crawl(keyword=query, max_num=n_per_query, min_size=(100, 100))
            except Exception as e:
                st.session_state["boost_download_error"] = str(e)
            finally:
                # Only signal completion — image scanning happens on the main thread
                # to avoid a race condition where the rerun fires before this write lands.
                st.session_state["boost_downloading"]   = False
                st.session_state["boost_download_done"] = True

        threading.Thread(
            target=_download_images,
            args=(boost_part_id, int(imgs_per_query), dl_root),
            daemon=True,
        ).start()
        st.rerun()

    # Polling spinner while download runs
    if st.session_state["boost_downloading"]:
        st.info("Downloading images from Bing… this may take 30–60 seconds.")
        time.sleep(1)
        st.rerun()

    # Once the thread signals done, scan the directory on the main thread.
    # This avoids the race condition where st.rerun() fires before the background
    # thread finishes writing boost_images to session state.
    if st.session_state["boost_download_done"] and not st.session_state["boost_images"]:
        _scan_part = st.session_state["boost_current_part"]
        _dl_root   = REAL_IMAGES_DIR / _scan_part / "raw_downloads"
        # Recursive scan — catches any subdirectory structure icrawler uses
        _found = []
        if _dl_root.exists():
            for _ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"):
                _found += sorted(_dl_root.rglob(_ext))
        st.session_state["boost_images"] = [str(p) for p in _found]
        if _found:
            st.rerun()   # re-render now that images are loaded
        else:
            if not st.session_state["boost_download_error"]:
                st.session_state["boost_download_error"] = (
                    f"No images found under {_dl_root}. "
                    "Bing may have returned 0 results — try a different part ID."
                )

    if st.session_state["boost_download_error"]:
        st.error(f"Download error: {st.session_state['boost_download_error']}")

    # Reload button — always visible when a part ID is entered, survives page refresh
    if boost_part_id:
        _reload_root = REAL_IMAGES_DIR / boost_part_id / "raw_downloads"
        _col_rel, _ = st.columns([1, 3])
        with _col_rel:
            if st.button("Load / Reload images from disk", key="boost_reload"):
                _refound = []
                if _reload_root.exists():
                    for _ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"):
                        _refound += sorted(_reload_root.rglob(_ext))
                st.session_state["boost_images"]         = [str(p) for p in _refound]
                st.session_state["boost_current_part"]   = boost_part_id
                st.session_state["boost_idx"]            = 0
                st.session_state["boost_download_done"]  = True
                st.session_state["boost_download_error"] = "" if _refound else \
                    f"No images found in {_reload_root} — fetch them first."
                st.rerun()

    # --- Labeling Interface ---
    _images  = st.session_state["boost_images"]
    _idx     = st.session_state["boost_idx"]
    _c_part  = st.session_state["boost_current_part"]

    if _images:
        st.divider()

        # Summary row
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Downloaded",  len(_images))
        _m2.metric("Labeled",     st.session_state["boost_labeled"])
        _m3.metric("Skipped",     st.session_state["boost_skipped"])
        _m4.metric("Remaining",   max(0, len(_images) - _idx))

        if _idx >= len(_images):
            st.success(
                f"All {len(_images)} images processed for part **{_c_part}**!  "
                f"{st.session_state['boost_labeled']} labeled, "
                f"{st.session_state['boost_skipped']} skipped.  "
                f"Click **Prepare Dataset** in the Train tab to include them."
            )
        else:
            _img_path = Path(_images[_idx])
            # Indicate topside vs underside from the parent folder name
            _search_label = _img_path.parent.name.replace("_", " ").title()
            st.write(
                f"**Image {_idx + 1} of {len(_images)}** — "
                f"Part `{_c_part}` · Search: *{_search_label}*"
            )
            st.progress(_idx / len(_images))

            try:
                _pil     = Image.open(str(_img_path)).convert("RGB")
                _orig_w, _orig_h = _pil.size

                st.caption("Drag the red box (corners, edges, or interior) to "
                           "surround the LEGO brick, then click **Save & Next**.")

                # Per-image box state — persists across Streamlit reruns
                _box_key = f"bbox_{_c_part}_{_idx}"
                if _box_key not in st.session_state:
                    st.session_state[_box_key] = {"x1": 5.0, "y1": 5.0,
                                                  "x2": 95.0, "y2": 95.0}

                # Encode image as base64 JPEG for the component
                _buf = BytesIO()
                _pil.save(_buf, format="JPEG", quality=85)
                _b64 = base64.b64encode(_buf.getvalue()).decode()

                # Render drag-handles bounding box component
                _result = _bbox_editor(
                    img_b64=_b64,
                    box=st.session_state[_box_key],
                    key=f"bbox_comp_{_c_part}_{_idx}",
                )
                if _result is not None:
                    st.session_state[_box_key] = _result

                _box = st.session_state[_box_key]
                # Convert % → pixels
                _x1 = int(_box["x1"] / 100 * _orig_w)
                _y1 = int(_box["y1"] / 100 * _orig_h)
                _x2 = int(_box["x2"] / 100 * _orig_w)
                _y2 = int(_box["y2"] / 100 * _orig_h)

                _btn1, _btn2 = st.columns([1, 1])
                with _btn1:
                    _save = st.button("Save & Next ✓", type="primary",
                                      key=f"save_{_c_part}_{_idx}")
                with _btn2:
                    _skip = st.button("Skip — no brick here",
                                      key=f"skip_{_c_part}_{_idx}")

                if _save:
                    _out_img = REAL_IMAGES_DIR / _c_part / "images"
                    _out_lbl = REAL_IMAGES_DIR / _c_part / "labels"
                    _out_img.mkdir(parents=True, exist_ok=True)
                    _out_lbl.mkdir(parents=True, exist_ok=True)

                    _stem = f"real_{_c_part}_{_idx:04d}"
                    _dest = _out_img / (_stem + _img_path.suffix)
                    _pil.save(str(_dest))

                    _xc = (_x1 + _x2) / 2 / _orig_w
                    _yc = (_y1 + _y2) / 2 / _orig_h
                    _nw = (_x2 - _x1) / _orig_w
                    _nh = (_y2 - _y1) / _orig_h
                    # class_id 0 = placeholder; remapped to true ID at prepare time
                    _label = f"0 {_xc:.6f} {_yc:.6f} {_nw:.6f} {_nh:.6f}"
                    with open(_out_lbl / (_stem + ".txt"), "w") as _lf:
                        _lf.write(_label + "\n")

                    st.session_state["boost_labeled"] += 1
                    st.session_state["boost_idx"]     += 1
                    st.rerun()

                if _skip:
                    st.session_state["boost_skipped"] += 1
                    st.session_state["boost_idx"]     += 1
                    st.rerun()

            except Exception as _e:
                st.warning(f"Cannot load image: {_e}. Click Skip to continue.")
                if st.button("Skip (error)", key=f"err_{_c_part}_{_idx}"):
                    st.session_state["boost_skipped"] += 1
                    st.session_state["boost_idx"]     += 1
                    st.rerun()

    # --- Review labeled images ---
    if boost_part_id:
        _rev_img_dir = REAL_IMAGES_DIR / boost_part_id / "images"
        _rev_lbl_dir = REAL_IMAGES_DIR / boost_part_id / "labels"
        _rev_imgs = sorted(_rev_img_dir.glob("*")) if _rev_img_dir.exists() else []
        if _rev_imgs:
            st.divider()
            st.subheader(f"Labeled images — {boost_part_id} ({len(_rev_imgs)} saved)")
            _cols = st.columns(3)
            for _ri, _rp in enumerate(_rev_imgs):
                _rlbl = _rev_lbl_dir / (_rp.stem + ".txt")
                try:
                    _rim = Image.open(str(_rp)).convert("RGB")
                    _rw, _rh = _rim.size
                    if _rlbl.exists():
                        _draw = ImageDraw.Draw(_rim)
                        for _line in _rlbl.read_text().splitlines():
                            _parts = _line.strip().split()
                            if len(_parts) == 5:
                                _, _xc, _yc, _nw, _nh = map(float, _parts)
                                _rx1 = int((_xc - _nw / 2) * _rw)
                                _ry1 = int((_yc - _nh / 2) * _rh)
                                _rx2 = int((_xc + _nw / 2) * _rw)
                                _ry2 = int((_yc + _nh / 2) * _rh)
                                _lw  = max(3, _rw // 150)
                                _draw.rectangle([_rx1, _ry1, _rx2, _ry2],
                                                outline="red", width=_lw)
                    with _cols[_ri % 3]:
                        st.image(_rim, caption=_rp.name, use_container_width=True)
                except Exception:
                    pass
