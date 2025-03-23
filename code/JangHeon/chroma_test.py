from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 임베딩 모델 로딩 (저장할 때 사용했던 동일 모델)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# 2. Chroma DB 경로 지정
chroma_path = "data/chroma_construction_db_v3"  # 폴더 존재하는지 반드시 확인!

# 3. Chroma 벡터스토어 객체 다시 불러오기
vectorstore = Chroma(
    persist_directory=chroma_path,
    embedding_function=embedding_model
)

query = "고소작업 중 추락 위험이 있음에도 불구하고, 안전난간대, 안전고리 착용 등 안전장치가 미흡하였음."
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"결과 {i}")
    print("문단 내용:", doc.page_content[:300])
    print("메타데이터:", doc.metadata)

print("저장된 문서 수:", vectorstore._collection.count())