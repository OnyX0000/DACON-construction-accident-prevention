import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
import re  # 정규표현식 모듈

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0번 GPU만 노출됨

# 04_RAG.py 에서 load_vector_db 함수만 직접 구현
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_vector_db(db_dir: str) -> Chroma:
    """벡터 DB를 로드하는 함수"""
    embedding_model_name = "jhgan/ko-sbert-nli"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    try:
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embedding
        )
        print(f"벡터 DB 로드 완료: {db_dir} (총 문서 수: {vectordb._collection.count()})")
        return vectordb
    except Exception as e:
        print(f"벡터 DB 로드 오류: {str(e)}")
        return None

# 프롬프트 템플릿 정의
def get_prompt_template():
    return """
    ### 참고 문맥:
    {context}

    ### 지침: 당신은 건설 안전 전문가입니다.
    - 답변은 반드시 핵심 용어와 주요 문구만을 사용하여 작성해야 합니다.
    - 서론, 배경 설명, 부연 설명은 절대 포함하지 마세요.
    - 답변은 간결하고 명확하게 100토큰 내외로 간결하게 작성하세요.
    - 콤마(,) 등의 특수기호는 최대한 적게 사용하세요.
    
    - 올바른 답변의 어조의 예시는 다음과 같습니다:
    "호스 및 장비 고장 시 작업방법 및 절차 수립과 TBM 및 정기교육 시 적절한 개인보호구 착용 및 관련 사고사례 교육"
    
    - 질문이 구체적이지 않거나 참고 문맥의 정보가 부족한 경우 경우 다음과 같이 답변하세요:
    "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."

    ### 질문:
    {question}
    [/INST]
    """

# 메타데이터 필터링 로직이 포함된 문맥 검색 함수
def get_relevant_context(vectordb, query, k=5):
    # 유사도 점수를 포함하여 검색 수행
    results = vectordb.similarity_search_with_score(query, k=k)
    
    if not results:
        return ""
    
    # 가장 유사도가 높은 문서와 해당 유사도 점수
    top_doc, top_score = results[0]
    top_title = top_doc.metadata.get('title', '')
    print(f"가장 유사도가 높은 문서 제목: '{top_title}', 유사도: {1 - top_score:.4f}")
    
    filtered_docs = []
    
    for doc, score in results:
        doc_title = doc.metadata.get('title', '')
        doc_subtitle = doc.metadata.get('subtitle', '')
        
        # 제목이 동일한 문서만 포함
        if doc_title != top_title:
            continue
        
        # 소제목에 "목적" 패턴이 있는 경우 제외 (정확한 단어 매칭)
        if re.search(r'목\s*적', doc_subtitle):
            print(f"소제목 제외: '{doc_subtitle}' - 목적 관련 패턴 검출")
            continue
        
        filtered_docs.append(doc)
    
    # 필터링 후 문서가 없으면, 기본으로 top_doc 사용
    if not filtered_docs:
        filtered_docs = [top_doc]
    
    return "\n\n".join([doc.page_content for doc in filtered_docs])

def initialize_llm(model_name="gemma3:27b"):
    try:
        return Ollama(model=model_name, temperature=0)
    except Exception as e:
        print(f"LLM 초기화 실패: {e}")
        return None

def run_rag_inference():
    """
    [Part 1] RAG 추론을 실행하여 생성된 문장(답변)을 파일로 저장하는 부분
    (이후 임베딩 생성을 위해 중간 결과 파일을 생성)
    """
    test_path = "./code/SangGyeom/data/test.csv"
    vector_db_path = "./code/SangGyeom/db/vector_db"
    # 중간 결과 파일 (생성된 문장 저장) 경로 지정
    generated_file_path = "./code/SangGyeom/data/generated_sentences0323_2.csv"

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

        # 메타데이터 필터링 로직이 적용된 문맥 추출
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

    # 생성된 문장(답변) 파일로 저장
    generated_df = pd.DataFrame({
        "ID": test_df["ID"],
        "재발방지대책 및 향후조치계획": results
    })
    generated_df.to_csv(generated_file_path, index=False, encoding="utf-8-sig")
    print(f"생성된 문장 파일 저장 완료: {generated_file_path}")

def generate_embeddings_and_save():
    """
    [Part 2] 생성된 문장 파일을 열고, 임베딩을 생성 후 최종 제출 파일을 저장하는 부분
    """
    # 중간 결과 파일 경로 (생성된 문장이 저장된 파일)
    generated_file_path = "./code/SangGyeom/data/generated_sentences_2.csv"
    # 최종 제출 파일 경로
    submission_path = "./code/SangGyeom/data/submission0323_2.csv"

    # 생성된 문장 파일 읽기
    try:
        df = pd.read_csv(generated_file_path, encoding="utf-8-sig")
        print(f"생성된 문장 파일 로드 완료: {df.shape[0]} 행")
    except Exception as e:
        print(f"생성된 문장 파일 로드 오류: {e}")
        return

    # 임베딩 생성
    embedding_model = SentenceTransformer("jhgan/ko-sbert-sts")
    embeddings = embedding_model.encode(df["재발방지대책 및 향후조치계획"].tolist())
    print(f"임베딩 생성 완료! 형태: {embeddings.shape}")

    # 결과 저장 (생성된 문장과 임베딩을 결합)
    embedding_df = pd.DataFrame(embeddings, columns=[f"vec_{i}" for i in range(embeddings.shape[1])])
    final_df = pd.concat([df, embedding_df], axis=1)
    final_df.to_csv(submission_path, index=False, encoding="utf-8-sig")
    print(f"제출 파일 저장 완료: {submission_path}")

if __name__ == "__main__":
    # 1. RAG 추론 실행 및 생성된 문장 파일 저장
    run_rag_inference()
    # 2. 생성된 문장 파일을 바탕으로 임베딩 생성 후 최종 제출 파일 저장
    generate_embeddings_and_save()
