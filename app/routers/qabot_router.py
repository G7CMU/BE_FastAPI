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

# def search_qdrant(collection_name, post_id, question):
#     query_vector = embedding_model.encode(question, show_progress_bar=True, device=device)

#     # Tìm kiếm tất cả các điểm trong collection
#     search_results = qdrant_client.search(
#         collection_name=collection_name,
#         query_vector=query_vector,
#         limit=3,
#         with_payload=True
#     )

#     # Lọc kết quả để chỉ lấy những điểm có id trong payload trùng với post_id
#     filtered_results = [hit for hit in search_results if hit.payload.get("id") == post_id]
#     print(filtered_results)
#     if filtered_results:
#         context = " ".join([hit.payload["chunk"] for hit in filtered_results])
#         return context
#     else:
#         return "Không tìm thấy kết quả phù hợp."
    
def search_qdrant(
    collection_name: str,
    question: str,
    post_id: int,
    limit: int = 1,
):
    print(question)
    question_vector = embedding_model.encode([question])[0]

    query_filter = {"must": [{"key": "id", "match": {"value": post_id}}]} if post_id else None
    print(query_filter)
    try:
        search_results = qdrant_client.search(
            collection_name="post",  
            query_vector=question_vector,  
            query_filter={
                "must": [
                    {"key": "id", "match": {"value": post_id}}  
                ]
            },
            limit=1,  
            with_payload=True 
        )

        print('search', search_results)
        if search_results:
            context = " ".join([hit.payload.get("content", "") for hit in search_results])
            return context
        else:
            return None
    except Exception as e:
        print(f"Lỗi khi tìm kiếm trong Qdrant: {e}")
        return None

   
# Cấu hình tham số
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


def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

@router.post("/chatbot")
async def chatbot(request: ChatbotRequest):
    try:
        context = search_qdrant("post", request.question, request.idPost)
        if not context:
            return {"answer": "Không tìm thấy dữ liệu phù hợp."}
        print('context', context)
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