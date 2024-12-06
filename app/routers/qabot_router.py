from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import retrieval_qa
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.chatbot_request import ChatbotRequest
import torch  

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

def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01,
        device=device  
    )
    return llm


def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

@router.post("/chatbot")
async def chatbot(request: ChatbotRequest):
    try:
        context = search_qdrant("chatbot_collection", request.question)
        if not context:
            return {"answer": "Không tìm thấy dữ liệu phù hợp."}

        llm = load_llm(model_file)

        template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời thì nói không biết, 
                    đừng cố tạo ra câu trả lời:\n{context}\n<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
        prompt = create_prompt(template)

        qa_input = {"context": context, "question": request.question}

        answer = llm.invoke(prompt.format(**qa_input))

        return {"answer": answer}
    except Exception as e:
        print(f"Lỗi: {e}")
        return {"error": str(e)}