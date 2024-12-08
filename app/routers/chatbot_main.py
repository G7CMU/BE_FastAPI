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

qdrant_client = QdrantClient(host="localhost", port=6333)

model_file = "./saved_model/vinallama-7b-chat_q5_0.gguf"
model_name = "hothanhtienqb/mind_map_blog_model"
embedding_model = SentenceTransformer(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device: ',device)
router = APIRouter()

def search_qdrant(collection_name, query_text):
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

config = {
    "top_k": 40,                 # Top-k sampling
    "top_p": 0.95,               # Top-p sampling (nucleus sampling)
    "temperature": 0.8,          # Sampling temperature
    "repetition_penalty": 1.1,   # Penalty for repeated tokens
    "last_n_tokens": 64,         # Consider last N tokens for penalty
    "seed": -1,                  # Random seed for sampling
    "max_new_tokens": 256,       # Max tokens to generate
    "stop": None,                # Stop sequences
    "stream": False,             # Stream output or not
    "reset": True,               # Reset model state before generation
    "batch_size": 8,             # Batch size for token evaluation
    "threads": -1,               # Auto-detect threads
    "context_length": -1,        # Full context length
    "gpu_layers": 12            # Number of layers to run on GPU
}

def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01,
        config=config
    )
    return llm


from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk

stop_words = set(stopwords.words('english'))
def clean_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    text = re.sub(r"[^\w\s.,!?;:]", "", text)

    return text

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
        
        llm = load_llm(model_file)
        print('3')
        template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời thì nói không biết, 
                    đừng cố tạo ra câu trả lời:\n{context}\n<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
        prompt = create_prompt(template)

        qa_input = {"context": context, "question": request.question}

        answer = llm.invoke(prompt.format(**qa_input))

        return {"answer": answer}
    except Exception as e:
        print(f"Lỗi: {e}")
        return {"error": str(e)}