from app.models.sentence_transformer import load_model
from qdrant.qdrant_service import create_collection, insert_embeddings, search_embeddings
from concurrent.futures import ThreadPoolExecutor, as_completed

model_path = "C:\\Users\\Admin\\Desktop\\New folder\\New folder\\MindMap\\saved_model"
model = load_model(model_path)

def create_new_collection(collection_name):
    try:
        vector_size = model.get_sentence_embedding_dimension()
        create_collection(collection_name, vector_size)
        print(f"Collection '{collection_name}' tạo thành công: kích thước vector {vector_size}")
    except Exception as e:
        print(f"Lỗi khi tạo collection: {str(e)}")

# add data to collection
def store_data_in_qdrant(collection_name, titles, texts):
    try:
        wiki_embeddings = model.encode(texts)
        payloads = [{"title": title, "text": text} for title, text in zip(titles, texts)]
        insert_embeddings(collection_name, wiki_embeddings, payloads)
        print(f"Đã lưu thành công dữ liệu vào collection '{collection_name}'")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào Qdrant: {str(e)}")

def search_in_qdrant(collection_name, content):
    try:
        search_embedding = model.encode([content])[0]
        results = search_embeddings(collection_name, search_embedding)
        return results
    except Exception as e:
        print(f"Lỗi khi tìm kiếm trong Qdrant: {str(e)}")
        return None

# def store_data_in_qdrant(collection_name, titles, texts, max_workers=20, batch_size=1000):
#     try:
#         if len(titles) != len(texts):
#             raise ValueError("Danh sách titles và texts phải cùng lent")

#         wiki_embeddings = model.encode(texts)
#         payloads = [{"title": title, "text": text} for title, text in zip(titles, texts)]

#         batches = [(wiki_embeddings[i:i + batch_size], payloads[i:i + batch_size])
#                    for i in range(0, len(wiki_embeddings), batch_size)]

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = []
#             for batch_embeddings, batch_payloads in batches:
#                 futures.append(executor.submit(insert_embeddings, collection_name, batch_embeddings, batch_payloads))

#             for future in as_completed(futures):
#                 try:
#                     future.result()
#                 except Exception as e:
#                     print(f"Lỗi khi lưu batch: {str(e)}")

#         print(f"Đã lưu thành công {len(titles)} điểm vào collection '{collection_name}'")
#     except Exception as e:
#         print(f"Lỗi khi lưu dữ liệu vào Qdrant: {str(e)}")

def add_post_to_qdrant(collection_name, title, content):
    # Mã hóa nội dung bài post
    embedding = model.encode([content])[0]  # Mã hóa nội dung bài post

    # Tạo payload cho Qdrant
    payload = {"title": title, "content": content}

    # Thêm embeddings vào Qdrant
    insert_embeddings(collection_name, [embedding], [payload])