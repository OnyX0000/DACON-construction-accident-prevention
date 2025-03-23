import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
import importlib.util

# 벡터 DB 모듈 로드
module_path = os.path.join(os.path.dirname(__file__), "04_RAG.py")
spec = importlib.util.spec_from_file_location("rag_module", module_path)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)
load_vector_db = rag_module.load_vector_db

# 프롬프트 템플릿 정의
def get_prompt_template():
    return """
    ### 참고 문맥:
    {context}

    ### 지침: 당신은 건설 안전 전문가입니다.
    - 답변은 반드시 핵심 용어와 주요 문구만을 사용하여 작성해야 합니다.
    - 서론, 배경 설명, 부연 설명은 절대 포함하지 마세요.
    - 답변은 간결하고 명확하게 40글자 내외로 간결하게 작성하세요.
    
    - 올바른 답변의 어조의 예시는 다음과 같습니다:
    "호스 및 장비 고장 시 작업방법 및 절차 수립과 TBM 및 정기교육 시 적절한 개인보호구 착용 및 관련 사고사례 교육"
    
    - 질문이 구체적이지 않거나 참고 문맥의 정보가 부족한 경우 경우 다음과 같이 답변하세요:
    "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."

    ### 질문:
    {question}
    [/INST]
    """

def get_relevant_context(vectordb, query, k=5):
    docs = vectordb.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def initialize_llm(model_name="gemma3:27b"):
    try:
        return Ollama(model=model_name, temperature=0)
    except Exception as e:
        print(f"LLM 초기화 실패: {e}")
        return None

def run_rag_inference():
    test_path = "./code/SangGyeom/data/test.csv"
    vector_db_path = "./code/SangGyeom/db/vector_db"
    submission_path = "./code/SangGyeom/data/submission0323.csv"

    # 데이터 로드
    test_df = pd.read_csv(test_path, encoding='utf-8-sig')
    vectordb = load_vector_db(vector_db_path)
    llm = initialize_llm()

    if vectordb is None or llm is None:
        print("벡터 DB 또는 LLM 초기화 실패")
        return

    results = []
    template = get_prompt_template()

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 추론"):
        query = row["사고원인"]

        context = get_relevant_context(vectordb, query)
        prompt = template.format(context=context, question=query)

        try:
            answer = llm.invoke(prompt)
        except Exception as e:
            print(f"오류 발생(ID: {row['ID']}): {e}")
            answer = "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."
            
        if not answer:
            answer = "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."
        
        print("query:", query, "context:", context[:20], "...", context[-20:], "answer:", answer, sep='\n')
        results.append(answer)

    # 임베딩 생성
    embedding_model = SentenceTransformer("jhgan/ko-sbert-sts")
    embeddings = embedding_model.encode(results)

    # 결과 저장
    submission_df = pd.DataFrame({
        "ID": test_df["ID"],
        "재발방지대책 및 향후조치계획": results
    })
    embedding_df = pd.DataFrame(embeddings, columns=[f"vec_{i}" for i in range(embeddings.shape[1])])
    final_df = pd.concat([submission_df, embedding_df], axis=1)
    final_df.to_csv(submission_path, index=False, encoding="utf-8-sig")

    print(f"제출 파일 저장 완료: {submission_path}")

if __name__ == "__main__":
    run_rag_inference()
