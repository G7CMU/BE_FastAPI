# app/routers/sentiment.py
from fastapi import APIRouter, HTTPException
from app.models.sentiment_request import SentimentRequest
from app.services.sentiment_service import predict_emotion

router = APIRouter()

@router.post("/predict_emotion")
async def predict_emotion_route(request: SentimentRequest):
    try:
        predicted_emotion = predict_emotion(request.sentence)
        return {"sentence": request.sentence, "predicted_emotion": predicted_emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
