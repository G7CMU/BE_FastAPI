from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain.text_splitter import CharacterTextSplitter

qdrant_client = QdrantClient(host="localhost", port=6333)

def create_db_qdrant():
    raw_text = """Kính gửi Quý khách hàng thân thiết,Cảm ơn Quý khách đã luôn tin tưởng và đồng hành cùng Vinaphone 
    trong suốt thời gian qua. Nhằm nâng cao chất lượng dịch vụ, chúng tôi xin gửi đến Quý khách những thông tin quan
      trọng trong bài viết này. Hãy dành chút thời gian đọc kỹ để cập nhật đầy đủ!1. Chương Trình Khuyến Mại Mới Nạp
        Thẻ, Nhận Ngay Ưu Đãi Siêu Hấp DẫnTừ ngày 10122024 đến 20122024, khi Quý khách nạp thẻ mệnh giá từ 100.000 
        VNĐ trở lên, sẽ được:Tặng ngay 20 giá trị nạp vào tài khoản chính.Nhận 1 GB dữ liệu miễn phí sử dụng trong 
        7 ngày.2. Điều Chỉnh Gói Cước Dữ Liệu 4GĐể đáp ứng nhu cầu sử dụng Internet di động ngày càng cao, Vinaphone 
        xin thông báo điều chỉnh một số gói cước như sau:Gói MAX100: 100.000 VNĐtháng, dung lượng 10 GB, miễn phí truy 
        cập các ứng dụng phổ biến như Zalo, Facebook.Gói MAX200: 200.000 VNĐtháng, dung lượng 30 GB, không giới hạn tốc
          độ truy cập.Lưu ý: Các thay đổi sẽ áp dụng từ ngày 01012025.3. Cảnh Báo Tin Nhắn Lừa ĐảoGần đây, Vinaphone ghi
            nhận nhiều trường hợp khách hàng nhận được tin nhắn mạo danh chúng tôi để lừa đảo. Để đảm bảo an toàn thông 
            tin, Quý khách vui lòng:Không cung cấp mã OTP, thông tin cá nhân qua bất kỳ đường dẫn lạ nào.Kiểm tra thông 
            tin chính thức trên website: www.vinaphone.com.vnNếu phát hiện tin nhắn lừa đảo, vui lòng liên hệ tổng đài
              18001091 để được hỗ trợ.4. Đổi SIM Miễn Phí Để Sử Dụng eSIMVinaphone đang triển khai chương trình đổi SIM
                vật lý sang eSIM miễn phí tại các cửa hàng Vinaphone trên toàn quốc.eSIM giúp Quý khách dễ dàng chuyển 
                đổi mạng, sử dụng nhiều số điện thoại trên cùng một thiết bị.Hạn đăng ký: đến hết ngày 31122024.5. 
                Thông Tin Về Giờ Hoạt Động Của Tổng ĐàiĐể phục vụ khách hàng tốt hơn, chúng tôi đã mở rộng thời gian hoạt
                  động của tổng đài chăm sóc khách hàng:Tổng đài 18001091: Hỗ trợ 247 mọi thắc mắc về dịch vụ.Tổng đài 
                  9191: Miễn phí cho thuê bao nội mạng, hoạt động từ 7h 22h hàng ngày.6. Triển Khai Công Nghệ 5GVinaphone
                    tự hào là một trong những nhà mạng đầu tiên triển khai dịch vụ 5G tại Việt Nam. Hiện tại, công nghệ 5G đã được phủ sóng tại:Hà Nội, Hồ Chí Minh, Đà Nẵng, và các khu vực trung tâm khác.Vinaphone luôn nỗ lực mang đến dịch vụ tốt nhất và trải nghiệm tuyệt vời cho Quý khách. Đừng quên theo dõi các chương trình ưu đãi và cập nhật thông tin mới nhất từ chúng tôi qua:Website: www.vinaphone.com.vnFanpage: Vinaphone OfficialNếu Quý khách có bất kỳ câu hỏi nào, hãy liên hệ ngay tổng đài 18001091 miễn phí để được hỗ trợ.Trân trọng cảm ơn!Vinaphone Kết nối mọi hành trình.
""" 

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_text(raw_text)

    embedding_model = SentenceTransformer("hothanhtienqb/mind_map_blog_model")
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    qdrant_client.recreate_collection(
        collection_name="chatbot_collection",
        vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine"),
    )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"chunk": chunks[i]})
        for i in range(len(chunks))
    ]
    qdrant_client.upsert(collection_name="chatbot_collection", points=points)
    print("Dữ liệu đã được lưu vào Qdrant!")

create_db_qdrant()


