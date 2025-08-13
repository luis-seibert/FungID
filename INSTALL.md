# Installation and Setup Guide

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/luis-seibert/FungID.git
   cd FungID
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv fungid_env
   source fungid_env/bin/activate  # On Windows: fungid_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Option A) Run the notebooks**
   ```bash
   jupyter lab
   ```

5. **(Option B) Launch the FastAPI server (with Grad-CAM heatmaps)**
   ```bash
   uvicorn fungid.backend.main:app --reload --port 8000
   ```
   Open http://localhost:8000/ for the drag & drop UI, or http://localhost:8000/docs for the OpenAPI schema.

## Hardware Requirements

- **GPU**: Not required, but recommended (CUDA-compatible for faster training)

## Dataset Download

The project uses community-sourced images from MushroomObserver. Run `01_dataset_acquisition.ipynb` to build the local catalog and download filtered species.

Notes:
- Approximate size: ~250 MB (depends on filtering)
- Total elapsed scrape time: 60–90 minutes (network dependent)

## GPU Setup (Optional)

For GPU acceleration:

1. Install CUDA-compatible PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in training configuration
   - Use CPU instead of GPU if memory limited

2. **Download Failures**
   - Check internet connection
   - Some images may be unavailable from MushroomObserver

3. **Missing Dependencies**
   - Ensure all packages from requirements.txt are installed
   - Update pip: `pip install --upgrade pip`

## Project Structure (Condensed)

```
FungID/
├── fungid/
│   ├── backend/            # FastAPI + static UI (Grad-CAM enabled)
├── utils/                  # Training & classic inference utilities
├── data/                   # CSV splits, images (after acquisition)
├── checkpoints/            # Saved model weights
├── 01..05_*.ipynb          # End-to-end workflow notebooks
├── requirements.txt
├── INSTALL.md
└── README.md
```

> Add `fungid/backend/static/preview_screenshot.png` manually if you want a README UI preview.

## Grad-CAM Output

Every prediction via `/predict` returns a `heatmap_overlay` base64 string (PNG). The web UI renders a side‑by‑side original vs. overlay comparison.

To decode programmatically:
```python
import base64, io
from PIL import Image
img = Image.open(io.BytesIO(base64.b64decode(heatmap_overlay)))
```
