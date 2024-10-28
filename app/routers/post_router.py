from fastapi import APIRouter, HTTPException
from app.models.post_request import PostRequest
from app.services.qdrant_service import add_post_to_qdrant, store_data_in_qdrant, create_new_collection

router = APIRouter()
from app.models.store_request import StoreRequest

@router.post("/add_post")
async def add_post(request: PostRequest):
    try:
        add_post_to_qdrant(request.collection_name , request.title, request.content)
        return {
             "message": f"Thêm post vào collection {request.collection_name} thành công"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
