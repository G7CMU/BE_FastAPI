from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware  # Import CORSMiddleware

from app.routers.search_router import router as search_router
from app.routers.qdrant_router import router as qdrant_router
from app.routers.post_router import router as post_router
from app.routers.sentiment_router import router as sentiment_router
from app.routers.suggestions_router import router as suggest_router
from app.routers.qabot_router import router as chatbot_router
from app.routers.test_db import router as add_post_qdrant
from app.routers.qabot_1post import router as chatbot_1post
from app.routers.chatbot_main import router as chatbot_main
from app.services.qdrant_service import store_data_in_qdrant
from app.core.load_model import load_model  # Giả định bạn có service này
from app.core.load_model import get_embedding_model

data_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Khởi động ứng dụng...")

    # Tải mô hình AI
    get_embedding_model()
    load_model()  # Đảm bảo mô hình LLM cũng được tải đúng cách
    print("Mô hình đã được tải thành công.")

    yield  # Chờ ứng dụng chạy

    # Đóng tài nguyên khi ứng dụng tắt
    print("Đóng ứng dụng...")

# Tạo ứng dụng FastAPI với lifecycle
app = FastAPI(lifespan=lifespan)

# Cấu hình CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi nguồn gốc, có thể thay bằng danh sách các nguồn gốc cụ thể
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các headers
)

# Đăng ký các router
app.include_router(search_router)
app.include_router(qdrant_router)
app.include_router(post_router)
app.include_router(sentiment_router)
app.include_router(suggest_router)
app.include_router(chatbot_router)
app.include_router(add_post_qdrant)
app.include_router(chatbot_1post)
app.include_router(chatbot_main)

# Các cấu hình khác
