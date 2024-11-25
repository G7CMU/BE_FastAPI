from fastapi import FastAPI
from app.routers.search_router import router as search_router
from app.routers.qdrant_router import router as qdrant_router
from app.routers.post_router import router as post_router
from app.routers.sentiment_router import router as sentiment_router
from app.routers.suggestions_router import router as suggest_router
from app.services.qdrant_service import search_in_qdrant, store_data_in_qdrant

app = FastAPI()

# Biến toàn cục để kiểm tra xem dữ liệu đã được tải chưa
data_loaded = False
# from datasets import load_dataset

# # Tải dataset từ Wikipedia
# dataset = load_dataset("wikipedia", "20220301.en", split="train")

# print(f"Số lượng item trong dataset: {len(dataset)}")

# @app.on_event("startup")
# async def startup_event():
#     global data_loaded
#     if not data_loaded:
#         titles, texts = load_wikipedia_data()
#         store_data_in_qdrant(titles, texts)
#         data_loaded = True  # Đánh dấu rằng dữ liệu đã được tải
#         print("Dữ liệu Wikipedia đã được lưu vào Qdrant.")

app.include_router(search_router)
app.include_router(qdrant_router)
app.include_router(post_router)
app.include_router(sentiment_router)
app.include_router(suggest_router)
#Các route khác và cấu hình ở đây