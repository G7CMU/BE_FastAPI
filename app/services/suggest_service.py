from sentence_transformers import SentenceTransformer, util

model_name = "hothanhtienqb/mind_map_blog_model" 
model = SentenceTransformer(model_name)

search_history = [
    "hướng dẫn tối ưu hóa mindmap",
    "ứng dụng AI trong phân tích sơ đồ tư duy",
    "phân tích dữ liệu với mô hình AI",
    "cách viết bài blog hấp dẫn",
    "xây dựng sơ đồ cảm xúc từ dữ liệu",
    "cách kể chuyện trong blogging",
    "hướng dẫn viết bài chuẩn SEO năm 2024",
    "các công cụ hỗ trợ viết blog tốt nhất",
    "tích hợp sơ đồ tư duy trong blog",
    "tạo bài viết tương tác với sơ đồ AI",
    "phân tích sentiment trong nội dung blog",
    "cách vẽ sơ đồ nhà bằng phần mềm miễn phí",
    "làm thế nào để tạo sơ đồ tư duy sáng tạo",
    "công cụ tạo sơ đồ mindmap online miễn phí",
    "so sánh các nền tảng viết blog phổ biến",
    "cách AI hỗ trợ viết bài chuyên sâu",
    "tự động hóa sơ đồ tư duy với Python",
    "tạo bản đồ kiến thức với AI",
    "tối ưu hóa mindmap để học tập",
    "phân tích nội dung blog bằng machine learning"
]


def get_similar_sentences(query: str, top_k: int):
    history_embeddings = model.encode(search_history, convert_to_tensor=True)

    query_embedding = model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, history_embeddings)
    similarities = similarities[0].cpu().numpy() 

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

