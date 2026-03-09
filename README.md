# Lego-Model-Factory

A Streamlit app for generating synthetic LEGO training data with Blender, training YOLO11 object detection models, and running live webcam inference.

## Features

- **Generate** — Render synthetic LEGO part images via Blender + LDraw with randomised lighting and HDRI backgrounds
- **Inspect** — Browse and visualise generated images with YOLO bounding box overlays
- **Train** — Prepare datasets and train YOLO11 models directly from the UI
- **Test** — Run live webcam inference with trained models

## Prerequisites

- Python 3.10+
- [Blender 4.3+](https://www.blender.org/download/) installed
- [Studio 2.0](https://www.bricklink.com/v3/studio/download.page) installed (provides the LDraw parts library)

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/Lego-Model-Factory.git
cd Lego-Model-Factory

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit the paths at the top of `app.py` to match your system:

```python
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
LDRAW_ROOT  = r"C:\Program Files\Studio 2.0\ldraw"
```

## Running

```bash
# Make sure the virtualenv is active
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
├── app.py              # Streamlit application
├── blender_gen.py      # Blender headless rendering script
├── dataset.yaml        # YOLO dataset config (auto-generated)
├── requirements.txt    # Python dependencies
├── models/             # Trained model weights and metrics
├── datasets/           # Generated raw and YOLO-formatted data (gitignored)
└── hdri/               # HDRI environment maps (gitignored)
```

## James' Notes For Improving the Model

- More lighting randomisation in the renders.
- Negative Samples (images without brick in it)
- Real images from web
- Greyscale image
