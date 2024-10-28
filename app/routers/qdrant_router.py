from fastapi import APIRouter, HTTPException
from app.models.search_request import SearchRequest
from app.services.qdrant_service import search_in_qdrant, store_data_in_qdrant, create_new_collection

router = APIRouter()
from app.models.store_request import StoreRequest

@router.post("/store_wiki_data")
async def store_wiki_data(request: StoreRequest):
    try:
        new_collection = request.collection_name
        create_new_collection(new_collection)
        return {
             "message": f"Tạo thành công collection {request.collection_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
