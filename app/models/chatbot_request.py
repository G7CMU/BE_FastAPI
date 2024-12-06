from pydantic import BaseModel
class ChatbotRequest(BaseModel):
    # title: str  
    question: str  
