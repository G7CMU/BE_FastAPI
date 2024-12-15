from pydantic import BaseModel
class ChatbotRequest(BaseModel):  
    idPost: str
    question: str
      
