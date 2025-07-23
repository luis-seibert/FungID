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

4. **Run the notebooks**
   ```bash
   jupyter lab
   ```

## Hardware Requirements

- **GPU**: Not required, but recommended (CUDA-compatible for faster training)

## Dataset Download

The project uses the MushroomObserver dataset. The data acquisition notebook (`01_dataset_acquisition.ipynb`) will automatically download the required data when executed.

**Note**: The complete image dataset is approximately 250 MB. The download process may take 60-90 minutes.

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

## Project Structure

After setup, your directory should look like:

```
FungID/
├── data/                 # Dataset (downloaded automatically)
├── checkpoints/         # Model weights (created during training)
├── utils/              # Core functionality
├── *.ipynb            # Jupyter notebooks
├── requirements.txt   # Dependencies
└── README.md         # Main documentation
```
