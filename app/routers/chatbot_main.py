from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import retrieval_qa
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.models import PointStruct, VectorParams
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.chatbot_request import ChatbotRequest
import torch  
import requests
from app.core.clean_text import clean_text
from app.core.load_model import load_model
qdrant_client = QdrantClient(host="localhost", port=6333)
from app.core.load_model import get_embedding_model, get_llm_model  # Import các hàm getter

# model_file = "./saved_model/vinallama-7b-chat_q5_0.gguf"
# model_name = "hothanhtienqb/mind_map_blog_model"
# embedding_model = SentenceTransformer(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device: ',device)
router = APIRouter()



def search_qdrant(collection_name, query_text):
    embedding_model = get_embedding_model()  # Lấy biến toàn cục qua getter
  # Lấy instance của mô hình LLM

    if embedding_model is None:
        raise ValueError("Embedding model is not loaded properly.")
    query_vector = embedding_model.encode(query_text, show_progress_bar=True, device=device)

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )
    if search_results:
        context = " ".join([hit.payload["chunk"] for hit in search_results])
        return context 
    else:
        return "Không tìm thấy kết quả phù hợp."



def pushQd(token: str, post_id: str):
    headers = {
            "Authorization": f"{token}"  
        }

    # Lấy dữ liệu bài viết từ API 
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
    embedding_model = get_embedding_model()
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

def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

 
class PostRequest(BaseModel):
    idPost: str
    question: str  
    token: str

@router.post("/chatbot2")
async def chatbot2(request: PostRequest):
    print('3')
    try:
        pushQd(request.token, request.idPost)
        print('1')
        context = search_qdrant("chatbot_collection1", request.question)
        print('2')
        if not context:
            return {"answer": "Không tìm thấy dữ liệu phù hợp."}
        print('3')
        template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời thì nói không biết, 
                    đừng cố tạo ra câu trả lời:\n{context}\n<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
        prompt = create_prompt(template)

        qa_input = {"context": context, "question": request.question}
        llm_model = get_llm_model()
        answer = llm_model.invoke(prompt.format(**qa_input))

        return {"answer": answer}
    except Exception as e:
        print(f"Lỗi: {e}")
        return {"error": str(e)}