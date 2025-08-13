# FungID: Deep Learning for Mushroom Species Classification

A computer vision project that uses deep learning to classify and identify mushroom species, specifically focused on distinguishing between edible and potentially bitter/inedible boletes. This project addresses an important food safety challenge by helping identify *Tylopilus felleus* (bitter bolete), which can render a meal inedible.

**Note**: This project is intended for educational and research purposes. Always consult mycological experts for definitive mushroom identification before consumption.

## Problem Statement

Mushroom identification is crucial for foragers and food safety. This project tackles the binary classification problem of distinguishing:

- **Edible boletes**: *Boletus edulis* (porcini), *Imleria badia* (bay bolete)
- **Bitter bolete**: *Tylopilus felleus* (bitter bolete) - inedible and extremely bitter

Misidentification can ruin a whole meal even if only a single small bitter bolete is contained. Making accurate automated classifications therefore is a valuable additional tool for identifying the inedible bolete.

## Technical Approach

### Dataset
- **Source**: [MushroomObserver](https://mushroomobserver.org/) community database
- **Size**: 1,411 high-quality images across 3 species
- **Distribution**: 
  - *Boletus edulis*: 594 images
  - *Imleria badia*: 218 images  
  - *Tylopilus felleus*: 599 images
- **Split**: 70% training, 15% validation, 15% testing

### Model Architecture
- **Base Model**: ResNet18 with ImageNet pre-trained weights
- **Transfer Learning**: Fine-tuned final layers for binary classification
- **Input**: 299×299 RGB images
- **Output**: Binary classification (edible vs. bitter)

### Training Strategy
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with weight decay
- **Batch Size**: 256 (with gradient accumulation)
- **Learning Rate**: Adaptive with ReduceLROnPlateau
- **Early Stopping**: Based on validation recall
- **Metric Focus**: Recall optimization (minimize false negatives for safety)

## Results

The model achieves strong performance on the test set:

- **Recall**: Optimized for detecting bitter boletes (safety-critical)
- **AUROC**: Comprehensive binary classification performance
- **Model Selection**: Best checkpoint based on validation recall

*Detailed performance metrics and visualizations available in `04_model_evaluation.ipynb`*

## Usage

See `INSTALL.md` for details.

### Running the Pipeline

1. **Data Acquisition** (`01_dataset_acquisition.ipynb`)
   - Download dataset catalog from MushroomObserver
   - Filter species by image count thresholds
   - Scrape images for selected species

2. **Data Preparation** (`02_dataset_preparation.ipynb`)
   - Create train/validation/test splits
   - Generate binary classification labels
   - Export dataset splits as CSV files

3. **Model Training** (`03_model_training.ipynb`)
   - Configure ResNet18 architecture
   - Train with transfer learning
   - Monitor metrics and save checkpoints

4. **Model Evaluation** (`04_model_evaluation.ipynb`)
   - Load best model checkpoint
   - Generate performance metrics
   - Create confusion matrix and ROC curves

5. **Inference** (`05_model_inference.ipynb`)
   - Interactive image classification
   - Real-time prediction interface
   - Confidence scoring

### Quick Inference
```python
from fungid.utils import image_classification

result = image_classification(
    image_path="path/to/mushroom.jpg",
    model_path="checkpoints/ResNet18_best_model.pth",
    model_name="ResNet18",
    number_classes=2,
    class_labels={0: "Edible Bolete", 1: "Bitter Bolete"}
)

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Project Structure

```
FungID/
├── fungid/                         # Python package root
│   ├── backend/                    # FastAPI application
│   │   └── main.py                 # API entrypoint (uvicorn fungid.backend.main:app)
│   └── utils/                      # Core utility modules
│       ├── dataset_acquisition.py  # Data downloading and scraping
│       ├── dataset_preparation.py  # Data splitting utilities
│       ├── image_dataset.py        # PyTorch dataset and transforms
│       ├── model_training.py       # Training loop and model utilities
│       ├── model_inference.py      # Inference and prediction functions
│       └── logger.py               # Logging configuration
├── 01_dataset_acquisition.ipynb    # Data collection and catalog processing
├── 02_dataset_preparation.ipynb    # Dataset splitting and preprocessing  
├── 03_model_training.ipynb         # Model training and optimization
├── 04_model_evaluation.ipynb       # Performance evaluation and metrics
├── 05_model_inference.ipynb        # Interactive classification interface
├── data/                           # Dataset and splits
│   ├── images/                     # Raw mushroom images by species
│   ├── *.csv                       # Train/validation/test splits
│   └── *.tsv                       # Species catalogs and metadata
├── checkpoints/                    # Model weights and training checkpoints
└── requirements.txt                # Python dependencies
```

## Possible Future Improvements

- Expand to multi-class classification with more species
- Implement ensemble methods for improved robustness
- Add explainability features (GradCAM, LIME)
- Mobile deployment for field use
- Integration with geographic/seasonal data

## Acknowledgments

- [MushroomObserver](https://mushroomobserver.org/) community for providing the dataset
- ImageNet pre-trained models for transfer learning foundation
- PyTorch ecosystem for deep learning framework

---