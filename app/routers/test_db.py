from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain.text_splitter import CharacterTextSplitter
import requests
from app.services.qdrant_service import create_new_collection
qdrant_client = QdrantClient(host="localhost", port=6333)
embedding_model = SentenceTransformer("hothanhtienqb/mind_map_blog_model")
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=50, length_function=len)

router = APIRouter()

class PostRequest(BaseModel):
    id: str
    token: str  

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(html_content: str) -> str:
    """
    Tiền xử lý nội dung HTML: loại bỏ thẻ HTML và các ký tự không cần thiết.
    
    Args:
        html_content (str): Nội dung HTML cần xử lý.
    
    Returns:
        str: Nội dung đã được xử lý.
    """

    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    
    text = re.sub(r"[^\w\s.,!?;:]", "", text)

    return text

def process_post_data(post_id: str, token: str):
    try:
        headers = {
            "Authorization": f"{token}" 
        }

        response = requests.get(f"https://api.khoav4.com/post/{post_id}", headers=headers)
        response.raise_for_status() 
        post_data = response.json()
 
        body = post_data.get("body", "")
        body = clean_text(body)
        print(body)
        if not body:
            raise ValueError("No 'body' field in post data")

        text_splitter = CharacterTextSplitter(separator=".", chunk_size=512, chunk_overlap=50, length_function=len)
        chunks = text_splitter.split_text(body)

        embedding_model = SentenceTransformer("hothanhtienqb/mind_map_blog_model")
        embeddings = [embedding_model.encode(chunk) for chunk in chunks]

        qdrant_client.recreate_collection(
            collection_name="chatbot_collection1",
            vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine"),
        )

        points = [
            PointStruct(id=i, vector=embeddings[i], payload={"chunk": chunks[i]})
            for i in range(len(chunks))
        ]
        qdrant_client.upsert(collection_name="chatbot_collection1", points=points)
        print("Dữ liệu đã được lưu vào Qdrant!")

        return {"message": "Data successfully processed and stored in Qdrant!"}
    except Exception as e:
        raise Exception(f"Error processing post data: {str(e)}")


@router.post("/processPost")
async def process_post_route(request: PostRequest):
    try:
        result = process_post_data(request.id, request.token)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
