import pandas as pd
import os
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import torch

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
sbert_model = SentenceTransformer("jhgan/ko-sbert-sts")
sbert_model = sbert_model.to(device)
vectorstore = FAISS.load_local(
    folder_path="/home/wanted-1/potenup-workspace/Project/dacon/DACON-construction-accident-prevention/code/JaeSik/db/20250323.faiss",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
llm = Ollama(model="gemma3:27b", temperature=0)

# ✅ 3. Few-shot Prompt (문장 종결 강조)
prompt_template = PromptTemplate(
    input_variables=["context", "question"] + metadata_columns,
    template="""\
너는 건설 안전 전문가야. 아래 사고원인에 대해 참고 문서를 기반으로 한 줄짜리 문장으로 끝나는 재발 방지 대책을 작성해.
**항상 완전한 문장(종결어미로 끝나는 문장) 형태**로 출력해줘. ('~다' 또는 '~습니다'로 끝나는 문장)

### 예시:
사고원인: 고소작업 중 추락 위험이 있음에도 불구하고, 안전난간대, 안전고리 착용 등 안전장치가 미흡하였음.
대책: 고소작업 시 추락 위험이 있는 부위에 안전장비를 설치해야 한다.

사고원인: 부주의
대책: 작업자에게 안전 교육을 정기적으로 시행해야 한다.

사고원인: 3층 슬라브 작업시 이동중 미끄러짐
대책: 이동 경로의 자재를 정리하고 미끄럼 방지 조치를 시행해야 한다.

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
    input_dict = {col: safe_get(row, col) for col in metadata_columns}

    try:
        # 문서 검색 + OR 조건 필터링
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
        vector = sbert_model.encode(answer, device=device).tolist()

        result_row = {
            "answer": answer
        }
        for i in range(768):
            result_row[f"vec_{i}"] = vector[i]

    except Exception as e:
        result_row = {
            "answer": f"[ERROR] {str(e)}"
        }
        for i in range(768):
            result_row[f"vec_{i}"] = None

    results.append(result_row)

# ✅ 6. 저장 (ID는 원본 index 기준, 사고원인 제외)
os.makedirs("submission", exist_ok=True)

submission_rows = []
for i, (idx, row) in enumerate(zip(df.index, results)):
    submission_row = {
        "ID": f"TRAIN_{idx}",
        "재발방지대책 및 향후조치계획": row["answer"]
    }
    for j in range(768):
        submission_row[f"vec_{j}"] = row.get(f"vec_{j}", None)

    submission_rows.append(submission_row)

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv("submission/final_submission_with_embedding.csv", index=False, encoding="utf-8-sig")
print("✅ 최종 제출 파일 저장 완료 → submission/final_submission_with_embedding.csv")
