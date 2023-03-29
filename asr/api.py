from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .Wav2vec2WithLM.model import Model, get_model

app = FastAPI()


class TranscriptionRequest(BaseModel):
    audio_path: str


class TranscriptionResponse(BaseModel):
    transcription: str

@app.post("/predict", response_model=TranscriptionResponse)
def predict(request: TranscriptionRequest, model: Model = Depends(get_model)):
    transcription = model.predict(request.audio_path)
    return TranscriptionResponse(transcription=transcription)
