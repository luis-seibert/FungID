from pydantic import BaseModel


class PredictionResponse(BaseModel):
    class_name: str
    is_bitter: bool
    confidence: float
    filename: str
    heatmap_overlay: str | None = None
    heatmap_error: str | None = None
