from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain.text_splitter import CharacterTextSplitter

qdrant_client = QdrantClient(host="localhost", port=6333)

def create_db_qdrant():
    raw_text = """Tất tần tật chuyện sau khi có con nhà tôi là như thế này .\nKhi vợ tôi bầu 33 tuần thì tôi đưa 
    vợ về nhà Nội để tiện chăm sóc khi sinh và sau sinh . Đến tuần 41 thì vợ sinh, mẹ vợ ở nhà Nội chăm vợ 15 
    ngày, tôi cũng ở nhà 15 ngày . Bố mẹ tôi thì vẫn còn trẻ nên vẫn đi làm , chỉ cơm nước và giặt giũ giúp thôi 
    vì vợ tôi cũng muốn là tự chăm theo cách mà cô muốn ( phương pháp Easy , tự ngủ ) và cô cũng không yên tâm 
    giao con cho ai cả . Nên vợ tự chăm mọi thứ từ con. Ở nhà Nội được 1 tháng mấy ngày thì vợ nhắn tôi muốn về 
    ngoại, tôi vẫn đi làm trong kia , không thể về ngay được nên tôi sắp xếp gọi xe đưa vợ về, mẹ tôi cũng đưa
      vợ về . Vợ về được 1 tuần thì tôi bay ra , lúc ấy tôi tôn trọng vợ ở ngoại hay cùng tôi vào trong đều được,
        tôi cũng chia sẻ thẳng thật với cô ấy là vào trong kia cô ấy sẽ rất vất vả , tôi thì đi làm suốt ngày ,
          vợ thì ở nhà một mình , vì ở nhà bố mẹ cô ấy rất chiều , cô ấy chỉ chăm con, bố mẹ vợ thì nghỉ hưu rồi
            nên có thời gian, nhiều lúc ăn sáng bố vợ cũng còn giục, bê vào tận phòng , mẹ cô ấy thì bế con cho . 
            Tất nhiên khác với ở nhà Nội, vì sáng bố mẹ tôi đi làm sớm nên chỉ để đồ ăn ở phòng cho cô ấy được 
            thôi, trưa tối về nấu cơm gọi cô ấy xuống ăn rồi thu dọn còn đi làm, tối thì mẹ tôi ngủ vố cô ấy 
            nhưng chỉ là ngủ thôi còn mọi việc vợ vẫn tự làm, tôi xem cam phòng nên biết . Lúc này thì quan hệ 
            mẹ con vẫn hoà thuận , vì vợ tôi cũng biết điều và mẹ tôi cũng thích vợ tôi . Hai mẹ con vẫn hoà 
            thuận, tôi cũng không thấy vợ tôi nói gì mẹ cả . Mẹ tôi thì trừ lúc tôi ở nhà thì phàn nàn về việc 
            chăm con theo cách của vợ là để con nằm cũi, và không cho ai bế ra , rồi tồi mẹ lên bế thì lại gọi 
            chồng ra bảo mẹ , ngoài chuyện ấy ra thì không có vấn đề gì. Sau khi ở nhà Ngoại 1 tuần thì vợ chồng 
            tôi, con , có cả mẹ tôi vào Nam cùng ( lúc đầu thì dự định chỉ có 2 vợ chồng cùng con nhưng vì bà nội
              nhớ cháu đêm không ngủ được nên tôi cũng đưa mẹ tôi đi cùng cho giải toả tâm lý). Mẹ tôi thì ở lại
                được 1 tuần rồi cũng về . Sau đó thì vợ chồng tôi chiến dịch tự chăm con . Lúc đầu thì vợ chăm 
                con, tôi cơm nước . Tất cả công việc liên quan đến con là cô ấy tự chăm . Tôi thì lo ngày 2 bữa
                  cơm . Sáng thì gần như vợ tôi nhịn vì không dậy được và tôi cũng lười nấu . Sau 1,2 tuần như 
                  vậy thì cô ấy tìm người giúp việc . Người giúp việc làm được 15 ngày thì tự nghỉ . Lúc ấy tôi
                    phải đi làm rồi nên vợ lo hết việc nhà : nấu cơm, rửa bát thì có máy rửa bát, quần áo cả con 
                    lẫn nhà đều có máy giặt, chỉ giặt tay mấy cái con trớ hay dính phân con, tôi cũng phụ vợ việc 
                    phơi đồ, nhà thì có robot lau nhà, tôi chịu vấn đề thay nước nôi, lau dọn robot , nhà tôi 
                    không thiếu cái thứ máy gì cả , thương vợ nên tôi cũng cắn răng mua đủ hết cho vợ đỡ vất vì 
                    tôi cũng biết tôi chẳng giúp được mấy , nhà thì có 3 con chó , sau khi vợ sinh thì cũng thả 
                    nó ở tầng trệt , bếp thì ở đó luôn nên vợ nấu cơm rồi cho nó ăn luôn. Tôi làm việc và ngủ ở 
                    tầng 1, tầng 2,3 là của hai mẹ con . Tôi cũng bảo vợ thuê người giúp việc , nhưng vợ gạt đi 
                    , nói là thuê lại phải chỉ dạy, rồi trông coi họ, còn mệt hơn. Tôi thấy vợ kỹ tính quá. Thế 
                    là tôi mua đồ ăn để tủ,rồi cứ thế buổi tối vợ cho con ngủ rồi vợ chế biến , hôm sau ăn cả trưa, tôi thỉnh thoảng về sớm cũng phụ vợ nấu cơm (hôm nào vợ mệt quá thì nhắn tôi mua gì đó về ăn. Hai mẹ con cứ vậy ở nhà chăm nhau. Vì tính chất công việc nên thỉnh thoảng tôi phải đi công tác trong ngày, tối về thì mẹ con ngủ rồi . Vì có con nhỏ nên tôi không yên tâm vợ con ở nhà một mình nên chỉ nhận đi công tác trong ngày thôi, đi đi rồi về, về thì tôi ngủ tầng 2 , vơi cũng bảo tôi ngủ tầng 2 . Cuộc sống cứ vậy cho đến khi bé nhà tôi 6,7 tháng, bắt đầu ăn dặm. Vợ bắt đầu mệt, sau đó sáng không dậy nổi để nấu ăn cho con . Tôi lúc ấy cũng có thời gian ở nhà : lúc đầu tôi cũng cố gắng nấu ăn cho hai vợ chồng, đồ ăn của con thì cô ấy nấu và cho ăn , cô ấy bảo muốn cho con ăn kiểu BLW, nó bày bừa dọn phát mệt.\nđược tầm 1 tuần thì bắt đầu nhà tôi dã ra. Chuyện là cái nhà mà gia đình tôi đang ở là thuê , vợ chồng quyết định mua 1 căn khác để ở cố định nên toàn bộ tiền tiết kiệm trước của tôi và cô ấy cũng dồn vào đấy . tôi vay bên nội 1 ít, người quen 1 ít trước Tết phải trả, vợ tôi cũng vay bên ngoại 1 ít , vợ tôi bảo vợ tôi tự trả khoản vay bên ngoại, tôi cũng bảo tôi vẫn lo được cho gia đình , để tôi trả thì vợ bảo để cô ấy trả, không đáng bao nhiêu so với tiền mua nhà , cô ấy nói dù không đáng bao nhiêu cô ấy cũng muốn góp một ít. vì cô ấy vẫn đang làm việc được. mỗi cây mỗi hoa, tài chính mỗi nhà mỗi khác, nhà tôi thì tôi kiếm tiền lớn lo trả nhưng chi phí nhiều như nhà cửa, ăn uống, đi lại … còn cô ấy thì lo chi phí của con và bản thân, tôi vẫn bảo trước khi sinh là cô ấy không lo được thì tôi vẫn lo được. cô ấy bảo cô ấy tự lập quen rồi nên cái gì lo được thì cô ấy lo. tính cô ấy vốn như vậy từ trước đến giờ nên tôi cũng không nói nhiều nhưng vợ tôi cần tiền tôi vẫn chuyển cho cô ấy . hàng tháng trả nợ người quen, nợ lãi nhà,tiền nhà thuê, sinh hoạt cũng hơn 150tr. cũng áp lực lắm các fen. nên tôi chăm chăm cố cày tiền còn vợ chăm con. cho đến khi con 6 tháng thì nà cử lộn xộn thì vợ kiệt sức,bắt đầu dậy muộn, cơm không muốn nấu, ăn không muốn ăn, nguy nhất là đồ ăn cho con cô ấy thỉnh thoảng còn quên, giờ giấc lung tung , mọi thứ đảo lộn , nát bét. Mà cô ấy kỹ con, đồ ăn con ăn nhất định cô ấy nấu, không cho mua bên ngoài … tôi thấy vậy nên bảo tôi đưa cô ấy về bắc cho có bố mẹ, ít ra thì có miếng cơm vào miệng. lúc đầu dự định đưa vợ về ngoại trước nhưng nhà Nội gần sân bay hơn nên là về Nội, về Nội được 2 tuần thì vợ tôi sáng này cũng ngủ dậy muộn, chỉ lo cơm nước cho con, lo cơm cho con thôi còn thấy cô ấy mệt ấy, bố mẹ tôi thì nấu cơm cho hai vơi chồng. vơi tôi từ trước vẫn là mồm miệng đỡ tay chân nên ông bà Nội cũng không mất lòng nên không nói gì, vẫn hoà hợp. Ở nhà đôi khi tôi cũng nấu cơm, lau nhà vì cũng ngại bố mẹ nói cô ấy lười. sáng nào bố mẹ tôi cũng dậy sớm bế cháu cho hai vợ ck ngủ. nhưng có hôm cô ấy 10 giờ vẫn chưa dậy, bố tôi bắt đầu kêu ca. vợ thì bảo là đêm con ti với dậy 3,4 lần nên 4,5 giờ cô mới ngủ mà 6h con đã dậy rồi. Bé nhà tôi thì trộm vía ai cũng bảo ngoan, mẹ tôi còn bảo cứ để ở nhà mẹ nuôi cũng được, ai bế cũng được, chỉ là nó nghịch không cho ngồi yên thôi, đêm thì nó có tỉnh giấc đòi ti hay đổi giấc thì vợ bế vác ôm nó . nhưng vợ thì kêu chăm con mệt, tôi không phản đối nhưng con tôi vậy là còn đỡ hơn nhiều rồi. Sau đó thì tôi đưa vợ về Ngoại 1 tuần còn tôi thì chạy về lo giấy tờ , thủ tục các kiểu . sau 1 tuần thì mẹ tôi làm cơm mời mọi người ở nhà nên vợ chồng tôi lại về Nội. về Nội được 5 ngày thì vợ tôi lại đòi về ngoại. tôi bảo ông bà Nội chưa chơi được cháu mấy đã đi rồi , vợ bảo thì Tết về chơi tiếp . rồi mọi chuyện vỡ ra khi mẹ tôi góp ý cách cho con ăn và ngủ của vợ tôi. Bà nấu hộ đồ ăn cho con, cho ít muối , vợ tôi không cho ăn, thói quen của bà hay cho muối vào nước dùng nên vợ tôi hổ cứ có muối là không cho ăn, mẹ nôi bón mồm cháu miếng giò hay gì thì trước mặt mẹ không nói gì, tối là càu nhàu với tôi . rồi ngủ thì cức cho con ngủ sớm , too với mẹ thì khuyên vợ cho ngủ lúc 8h tối hẵng cho ngủ nhưng vợ thì cứ muốn con ngủ sớm, nhiều lúc con khóc ré lên nghe mà bực với vợ, nó đã muốn ngủ đáu mà cho ngủ, mỗi lần vậy là bà lên bế con xuống, vợ thì càu nhàu bảo là nó gắt ngủ nên mới khóc, mọi người cứ vậy nó mất nếp, rồi bảo không hiểu gì cứ nghĩ nó không muốn ngủ … mẹ tôi bảo vợ tôi kỹ quá, các gì chăm con cũng nhất nhất theo ý vợ. ăn cơm chung thì góp ý với vợ, bà bảo ngày xưa nuôi có thế nào đâu vân khoẻ mạnh như trâu. vợ ko nói gì nhưng tối lại càu nhàu với tôi. rồi lại đòi về ngoại, bố mẹ tôi thì thương cháu muốn cháu ở đây. rồi vợ muốn nghỉ việc để chăm con, tôi ko ý kiến về tiền hay gì chỉ sợ vợ quanh quẩn xung quanh con, rồi khi con đi học ấy thì mất phương hướng, hiện giờ đã chỉ con là con, đồ cho con thì toàn giá cao , dù tiền của vợ nhưng tôi thấy nó cứ hơi quá, vợ quên mất cả bản thân mình, nhiều lúc tôi thấy làm quá mức, kỹ con quá . hôm tôi nhẹ nhàng nói suy nghĩ của tôi cho vợ thì vợ khóc lóc bảo từ lúc bầu đến bây giờ chưa bao giờ nghỉ làm , chưa phụ thuộc tôi chi phí bản thân hay cho con , bảo là về nhà để cho vợ được nghỉ ngơi mà ở đây cứ canh cánh phải dậy điểm danh, làm gì cũng phải để ý, công việc không làm được, con thì mọi người không giúp được nhưng cứ ý kiến với yêu cầu ( tôi thấy không phải mọi ng không giúp mà vợ kỹ tính quá nên ko ai dám giúp ) , ở đây thì không bạn bè người quen, ở trong kia thì hai mẹ con, ở đây thêm ông bà, khác gì như lồng nhỏ đến lồng to, về ngoại còn có bố mẹ, anh chị em, bạn bè, ng quen , còn đi gặp bạn, đi chơi, biết chỗ mà đi , đường xá quen thuộc… đỉnh điểm là hôm vợ cho con ngủ sớm tôi lớn tiếng bảo vợ thì vợ quát gắt cả tôi, bảo nó dụi mắt gãi tai nên mới cho ngủ chứ, rồi bà nấu đồ cho ăn từ cơm thì vợ nhất quyết nấu từ gạo bảo ko được nấu từ cơm nguội với khéo bà lại cho muối , tối bà với ông bế đi chơi đến 8 giờ trả vợ cho đi ngủ thì vợ hậm hực , tôi nghĩ chẳng mấy khi về nhà, bỏ qua được thì bỏ quá, vợ thì nhất quyết phải như vợ hay làm, cho con uống siro nó trớ mọi ng bảo đừng cho uống thì bảo nếu cái này nó ko uống khi nó ốm cái gì cũng ko ăn thì kệ nó à. nói chung ko ai đụng được . tôi bảo là vợ đùng kỹ con quá, tôi lo kinh tế, nhà cửa sắp xong , đấy là nhà mình không phải lo kinh tế nếu như ko có tiền thì chăm con như vợ có được không, tôi phải bảo bố mẹ vợ đừng chiều vợ quá chứ chiếu quá rồi có cái tính cái gì cũng theo ý mình thì mệt lắm . vợ khóc xong im lặng 3 ngày thì qua vợ thu xếp về ngoại, vợ gọi bảo bố vợ xin phép cho về ngoại, tôi gọi xe cho vợ. đêm qua thì vợ bảo vợ muốn ly thân, tết sẽ về nhưng từ giờ đến tết và sau tết sẽ ở bên ngoại. không cần bấy cứ gì của tôi chỉ cần con. tôi nghĩ vợ bốc đồng nên ko dám nói gì. chuyện là vậy""" 

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


