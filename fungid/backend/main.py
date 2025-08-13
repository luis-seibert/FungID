import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fungid.backend.model_inference import image_classification
from fungid.backend.models import PredictionResponse

app = FastAPI(title="FungID API", description="Bitter bolete classifier demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000/"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


MODEL_PATH = "checkpoints/ResNet18_best_model.pth"
MODEL_NAME = "ResNet18"
NUMBER_CLASSES = 2
CLASS_LABELS = {0: "Non-bitter Bolete (Edible)", 1: "Bitter Bolete (Unpalatable)"}


@app.get("/")
async def root_index() -> FileResponse:
    """Serve the main page.

    Returns:
        FileResponse: The index.html file from the static directory.

    Raises:
        HTTPException: If index.html is not found.
    """
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_file)


@app.post("/predict", response_model=PredictionResponse)
async def predict_bitter_bolete(file: UploadFile = File(...)) -> PredictionResponse:
    """Run image classification on the uploaded file.

    Args:
        file (UploadFile, optional): The uploaded image file. Defaults to File(...).

    Raises:
        HTTPException: If the uploaded file is not an image, is empty, or if inference fails.

    Returns:
        PredictionResponse: The classification result including class name, bitterness status, confidence, and filename.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            raise HTTPException(status_code=400, detail="Unsupported image format")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp.write(content)
            tmp_path = tmp.name

        classification_result = image_classification(
            tmp_path,
            MODEL_PATH,
            MODEL_NAME,
            NUMBER_CLASSES,
            CLASS_LABELS,
            return_heatmap=True,
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    return PredictionResponse(
        class_name=classification_result["class_name"],
        is_bitter=classification_result["is_bitter"],
        confidence=classification_result["confidence"],
        filename=file.filename
        or os.path.basename(classification_result.get("filename", "uploaded_image")),
        heatmap_overlay=classification_result.get("heatmap_overlay"),
        heatmap_error=classification_result.get("heatmap_error"),
    )
