from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain.text_splitter import CharacterTextSplitter
import requests

# Khởi tạo các thành phần cần thiết
qdrant_client = QdrantClient(host="localhost", port=6333)
embedding_model = SentenceTransformer("hothanhtienqb/mind_map_blog_model")
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=50, length_function=len)

# Định nghĩa router
router = APIRouter()

# Định nghĩa model để xác thực dữ liệu request
class PostRequest(BaseModel):
    id: str
    token: str  # Thêm trường token


# Hàm xử lý dữ liệu bài viết
def process_post_data(post_id: str, token: str):
    try:
        # Định nghĩa header với token
        headers = {
            "Authorization": f"{token}"  # Gửi token trong header
        }

        # Lấy dữ liệu bài viết từ API với header
        response = requests.get(f"https://api.khoav4.com/post/{post_id}", headers=headers)
        response.raise_for_status()  # Bắt lỗi nếu không thành công
        post_data = response.json()
        # Lấy nội dung bài viết
        body = post_data.get("body", "")
        print(body)
        if not body:
            raise ValueError("No 'body' field in post data")

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=50, length_function=len)
        chunks = text_splitter.split_text(body)

        embedding_model = SentenceTransformer("hothanhtienqb/mind_map_blog_model")
        embeddings = [embedding_model.encode(chunk) for chunk in chunks]

        qdrant_client.recreate_collection(
            collection_name="chatbot_collection",
            vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine"),
        )

        points = [
            PointStruct(id=i, vector=embeddings[i], payload={"chunk": chunks[i]})
            for i in range(len(chunks))
        ]
        qdrant_client.upsert(collection_name="chatbot_collection", points=points)
        print("Dữ liệu đã được lưu vào Qdrant!")

        return {"message": "Data successfully processed and stored in Qdrant!"}
    except Exception as e:
        raise Exception(f"Error processing post data: {str(e)}")


@router.post("/processPost")
async def process_post_route(request: PostRequest):
    try:
        result = process_post_data(request.id, request.token)  # Truyền thêm token vào
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
