from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models.suggest_request import SuggestRequest
from app.services.suggest_service import get_similar_sentences

router = APIRouter()

@router.websocket("/ws/suggest")
async def websocket_suggest(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
     
            data = await websocket.receive_json()
            query = data.get("query", "")
            top_k = data.get("top_k", 5)

            result = get_similar_sentences(query, top_k)
            
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.close()
        print(f"Error: {e}")
