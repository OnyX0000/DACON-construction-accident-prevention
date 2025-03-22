from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama

# 1. 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 2. Chroma DB 불러오기
vectorstore = Chroma(
    persist_directory="data/chroma_construction_db_v2",
    embedding_function=embedding_model
)

# 3. Ollama LLM 연결
llm = ChatOllama(model="gemma:27b", temperature=0)

# 4. 프롬프트 템플릿
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the best construction safety expert of the world.
Please write a **one-line countermeasure** for preventing recurrence by referring to the contents of the document below and the cause of the accident.
If you can't find the information from the PDF document, just say that you don't know.

Format Conditions:
- **One sentence**, **Centered on clear action**
- **Write in Korean**
- It's not too long, it's a short, actionable way to get into CSV
- Example: "Install safeguards in the fall hazard zone during complaint work"

Context of the document : {context}

Cause of the accident : {question}

answer:
"""
)

# 5. RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# 6. 사용자 입력 → 결과 생성
query = "타워크레인 해체 작업 중 작업자 간 소통 부족으로 안전조치가 누락되었음"
response = qa_chain.run(query)

print("AI의 재발 방지 대책:")
print(response)
