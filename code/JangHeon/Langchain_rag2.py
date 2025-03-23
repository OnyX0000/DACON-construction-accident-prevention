import pandas as pd
from tqdm import tqdm
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. CSV 로딩
csv_path = "data/cleaned_construction_accidents.csv"
df = pd.read_csv(csv_path)

question_column = "사고원인"
reference_column = "재발방지대책 및 향후조치계획"
metadata_columns = ["공사_대분류", "공종_세부", "작업프로세스", "인적사고", "물적사고"]
df = df.dropna(subset=[question_column] + metadata_columns)

# 2. 모델 로딩
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
vectorstore = Chroma(
    persist_directory="data/chroma_construction_db_v2",
    embedding_function=embedding_model
)
llm = Ollama(model="gemma3:27b", temperature=0)
sbert_model = SentenceTransformer("jhgan/ko-sbert-sts")

# 3. 프롬프트 템플릿
prompt_template = PromptTemplate(
    input_variables=["context", "question", "공사_대분류", "공종_세부", "작업프로세스", "인적사고", "물적사고"],
    template="""
    ### 참고 문맥:
    {context}
    
    ### 지침: 당신은 건설 안전 전문가입니다.
    답변은 간결하고 명확하게 최대 64 토큰 이내로 간결하게 작성하세요.
    서론, 배경 설명, 부연 설명은 절대 포함하지 마세요.
    질문이 구체적이지 않거나 참고 문맥의 정보가 부족한 경우 경우 다음과 같이 답변하세요:
    "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."
    
    ### 질문:
    {question}

    [/INST]
"""
)

# 4. 결과 리스트 초기화
results = []

def safe_get(row, col):
    val = row.get(col, "")
    return "" if pd.isna(val) else str(val)

# 5. 추론 루프
for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating single-action countermeasures")):
    question = row[question_column]
    reference = str(row.get(reference_column, "")).strip()
    similarity_score = None

    input_dict = {
        "question": question,
        "공사_대분류": safe_get(row, "공사_대분류"),
        "공종_세부": safe_get(row, "공종_세부"),
        "작업프로세스": safe_get(row, "작업프로세스"),
        "인적사고": safe_get(row, "인적사고"),
        "물적사고": safe_get(row, "물적사고")
    }

    try:
        # 문서 검색 및 필터링
        docs = vectorstore.as_retriever(search_kwargs={"k": 50}).get_relevant_documents(question)
        filtered_docs = [
            doc for doc in docs if (
                doc.metadata.get("공사_대분류") == input_dict["공사_대분류"] or
                doc.metadata.get("공종_세부") == input_dict["공종_세부"] or
                doc.metadata.get("작업프로세스") == input_dict["작업프로세스"] or
                doc.metadata.get("인적사고") == input_dict["인적사고"] or
                doc.metadata.get("물적사고") == input_dict["물적사고"]
            )
        ]

        # 문맥 구성 (상위 3개)
        top_context = "\n\n".join([doc.page_content for doc in filtered_docs[:3]])

        # LLM 호출
        final_prompt = prompt_template.format(context=top_context, **input_dict)
        answer = llm.invoke(final_prompt).strip()

        # 유사도 계산
        if reference and answer and not answer.startswith("[ERROR]"):
            emb1 = sbert_model.encode(answer, convert_to_tensor=True)
            emb2 = sbert_model.encode(reference, convert_to_tensor=True)
            similarity_score = cos_sim(emb1, emb2).item()

    except Exception as e:
        answer = f"[ERROR] {str(e)}"

    results.append({
        "사고원인": question,
        "answer": answer,
        "유사도": similarity_score,
        **{key: input_dict[key] for key in metadata_columns}
    })

    # 샘플 출력 (앞 3개만)
    if i < 3:
        print("\n샘플 결과 ------------------------------")
        print(f"사고원인: {question}")
        print(f"원문 대책: {reference}")
        print(f"생성된 대책: {answer}")
        if similarity_score is not None:
            print(f"유사도 (cosine): {similarity_score:.4f}")
        else:
            print("유사도: 계산 불가")
        print("----------------------------------------\n")

# 6. 저장
output_dir = "code/JangHeon/result"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "final_rag_answers_single_line.csv")
pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 저장 완료 → {output_path}")
