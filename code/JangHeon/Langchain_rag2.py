import pandas as pd
import os
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# ✅ 매핑: 원본 metadata 컬럼명 → FAISS 내 키 이름
metadata_mapping = {
    "공사_대분류": "construction_type",
    "공종_세부": "work_type",
    "작업프로세스": "work_process",
    "인적사고": "accident_type",
    "물적사고": "accident_object"
}

# ✅ 1. 데이터 로드
csv_path = "data/test_with_metadata.csv"
df = pd.read_csv(csv_path)

question_column = "사고원인"
reference_column = "재발방지대책 및 향후조치계획"
metadata_columns = list(metadata_mapping.keys())
df = df.dropna(subset=[question_column] + metadata_columns)

# ✅ 2. 모델 로딩
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
sbert_model = SentenceTransformer("jhgan/ko-sbert-sts")
vectorstore = FAISS.load_local(
    folder_path="code/Jaesik/20250323.faiss",
    embeddings=embedding_model
)
llm = Ollama(model="gemma3:27b", temperature=0)

# ✅ 3. Few-shot Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"] + metadata_columns,
    template="""
너는 건설 안전 전문가야. 아래 사고원인에 대해 참고 문서를 기반으로 한 줄짜리 재발 방지 대책을 작성해.

### 예시:
사고원인: 고소작업 중 추락 위험이 있음에도 불구하고, 안전난간대, 안전고리 착용 등 안전장치가 미흡하였음.
대책: 고소작업 시 추락 위험이 있는 부위에 안전장비 설치.

사고원인: 부주의
대책: 재발 방지 대책 마련과 안전교육 실시.

사고원인: 3층 슬라브 작업시 이동중  미끄러짐
대책: 현장자재 정리와 안전관리 철저를 통한 재발 방지 대책 및 공문 발송을 통한 향후 조치 계획.

--- 문서 내용 ---
{context}

--- 실제 사고원인 ---
{question}

대책:
"""
)

# ✅ 4. 유틸 함수
def safe_get(row, col):
    val = row.get(col, "")
    return "" if pd.isna(val) else str(val)

# ✅ 5. 생성 루프
results = []

for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating with Few-shot + Embedding")):
    question = safe_get(row, question_column)
    reference = safe_get(row, reference_column)

    input_dict = {col: safe_get(row, col) for col in metadata_columns}

    try:
        # 문서 검색 + OR 필터링
        docs = vectorstore.as_retriever(search_kwargs={"k": 50}).get_relevant_documents(question)
        filtered_docs = [
            doc for doc in docs if any(
                doc.metadata.get(metadata_mapping[key]) == input_dict[key]
                for key in metadata_columns
            )
        ]
        top_context = "\n\n".join([doc.page_content for doc in filtered_docs[:3]])

        # LLM 질의
        final_prompt = prompt_template.format(context=top_context, question=question, **input_dict)
        answer = llm.invoke(final_prompt).strip()

        # 후처리
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        if "," in answer:
            answer = answer.split(",")[0].strip()

        # 768차원 임베딩
        vector = sbert_model.encode(answer).tolist()

        result_row = {
            "사고원인": question,
            "answer": answer
        }
        for i in range(768):
            result_row[f"dim_{i}"] = vector[i]

    except Exception as e:
        result_row = {
            "사고원인": question,
            "answer": f"[ERROR] {str(e)}"
        }
        for i in range(768):
            result_row[f"dim_{i}"] = None

    results.append(result_row)

# ✅ 6. 저장
os.makedirs("submission", exist_ok=True)
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission/final_submission_with_embedding.csv", index=False, encoding="utf-8-sig")
print("✅ 최종 제출 파일 저장 완료 → submission/final_submission_with_embedding.csv")
