from pydantic import BaseModel

class SuggestRequest(BaseModel):
    query: str 
    top_k: int = 3  
