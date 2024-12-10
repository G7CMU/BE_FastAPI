from langchain_community.llms.ctransformers import CTransformers
from sentence_transformers import SentenceTransformer

model_file = "./saved_model/vinallama-7b-chat_q5_0.gguf"
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
    "gpu_layers": 12
}

llm_instance = None
embedding_model_instance = None

def load_model():
    global llm_instance
    if llm_instance is None:
        print("Đang tải model...")
        llm_instance = CTransformers(
            model="./saved_model/vinallama-7b-chat_q5_0.gguf",
            model_type="llama",
            max_new_tokens=1024,
            temperature=0.01,
            config=config
        )
        print("Model đã được tải.")
    return llm_instance

def get_llm_model():
    global llm_instance
    if llm_instance is None:
        load_model()  # Đảm bảo mô hình LLM được tải trước khi trả về
    return llm_instance


def get_embedding_model():
    global embedding_model_instance
    if embedding_model_instance is None:
        model_name = "hothanhtienqb/mind_map_blog_model"
        try:
            print(f"Đang tải mô hình embedding: {model_name}")
            embedding_model_instance = SentenceTransformer(model_name)
            print("Mô hình embedding đã được tải thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình embedding: {e}")
            return None
    return embedding_model_instance
