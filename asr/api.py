from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .Wav2vec2WithLM.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    audio_path: str


class SentimentResponse(BaseModel):
    transcription: str

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    transcription = model.predict(request.audio_path)
    return SentimentResponse(transcription=transcription)