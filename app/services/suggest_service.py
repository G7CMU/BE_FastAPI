from sentence_transformers import SentenceTransformer, util

model_name = "hothanhtienqb/mind_map_blog_model" 
model = SentenceTransformer(model_name)

search_history = [
    "hướng dẫn viết blog hiệu quả",
    "tạo sơ đồ tư duy bằng AI",
    "cách phân tích dữ liệu cảm xúc",
    "tối ưu hóa bài viết SEO",
    "sử dụng mô hình AI trong blogging",
    "tạo sơ đồ mindmap",
    "tạo sơ đồ nhà trường",
    "tạo sơ đồ yêu nhau"
]

def get_similar_sentences(query: str, top_k: int):
    # Mã hóa lịch sử tìm kiếm
    history_embeddings = model.encode(search_history, convert_to_tensor=True)

    # Mã hóa câu truy vấn
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Tính toán độ tương đồng cosine
    similarities = util.cos_sim(query_embedding, history_embeddings)
    similarities = similarities[0].cpu().numpy()  # Chuyển sang mảng numpy

    # Lấy các kết quả hàng đầu
    top_results = sorted(
        [(search_history[i], similarities[i]) for i in range(len(search_history))],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Chuyển đổi điểm tương đồng (score) sang float
    return {
        "suggestions": [
            {"text": result, "similarity": float(score)}  # Chuyển score thành float
            for result, score in top_results
        ]
    }

