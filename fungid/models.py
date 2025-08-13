from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str
    is_bitter: bool
    confidence: float
    filename: str