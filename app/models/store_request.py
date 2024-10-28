# app/models/store_request.py
from pydantic import BaseModel

class StoreRequest(BaseModel):
    collection_name: str

