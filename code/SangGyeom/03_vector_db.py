import os
import pandas as pd
import time
from tqdm import tqdm
from typing import List, Dict, Any

# 최신 LangChain API 사용
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """CSV 파일에서 데이터를 로드하는 함수
    
    Args:
        csv_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
        return df
    except Exception as e:
        print(f"데이터 로드 오류: {str(e)}")
        return pd.DataFrame()

def prepare_documents(df: pd.DataFrame) -> List[Document]:
    """데이터프레임을 Document 객체 리스트로 변환하는 함수
    
    Args:
        df (pd.DataFrame): 변환할 데이터프레임
        
    Returns:
        List[Document]: Document 객체 리스트
    """
    documents = []
    
    print("문서 변환 중...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="문서 준비"):
        # 필수 필드 확인
        if 'sentence' not in row or pd.isna(row['sentence']):
            continue
            
        # 메타데이터 구성
        metadata = {
            'title': row.get('title', ''),
            'subtitle': row.get('subtitle', ''),
            'page': row.get('page', ''),
            'line_num': row.get('line_num', ''),
            'source': row.get('source', '')
        }
        
        # Document 객체 생성 및 추가
        doc = Document(
            page_content=row['sentence'],
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"문서 준비 완료: {len(documents)}개")
    return documents

def create_vector_db(documents: List[Document], db_dir: str) -> Chroma:
    """Document 객체 리스트로 벡터 DB를 생성하는 함수
    
    Args:
        documents (List[Document]): Document 객체 리스트
        db_dir (str): 벡터 DB를 저장할 디렉토리 경로
        
    Returns:
        Chroma: 생성된 벡터 DB 객체
    """
    # 디렉토리 생성
    os.makedirs(db_dir, exist_ok=True)
    
    # Baseline.ipynb와 동일한 임베딩 모델 설정
    embedding_model_name = "jhgan/ko-sbert-nli"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    print(f"임베딩 모델 로드 완료: {embedding_model_name}")
    
    # 벡터 DB 생성 시작 시간 기록
    start_time = time.time()
    
    print(f"벡터 DB 생성 시작 ({len(documents)} 문서)...")
    
    # Chroma 벡터 DB 생성
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=db_dir
    )
    
    # 벡터 DB 생성 완료 시간 계산
    elapsed_time = time.time() - start_time
    
    # DB 저장
    vectordb.persist()
    print(f"벡터 DB 생성 및 저장 완료: {db_dir}")
    print(f"처리 시간: {elapsed_time:.2f}초 (평균 {elapsed_time/len(documents):.4f}초/문서)")
    
    return vectordb

def load_vector_db(db_dir: str) -> Chroma:
    """저장된 벡터 DB를 로드하는 함수
    
    Args:
        db_dir (str): 벡터 DB가 저장된 디렉토리 경로
        
    Returns:
        Chroma: 로드된 벡터 DB 객체
    """
    # Baseline.ipynb와 동일한 임베딩 모델 설정
    embedding_model_name = "jhgan/ko-sbert-nli"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Chroma 벡터 DB 로드
    vectordb = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding
    )
    
    print(f"벡터 DB 로드 완료: {db_dir}")
    print(f"총 문서 수: {vectordb._collection.count()}")
    return vectordb

def process_data_to_vector_db(input_csv: str, db_dir: str) -> None:
    """CSV 데이터를 처리하여 벡터 DB로 변환하는 함수
    
    Args:
        input_csv (str): 입력 CSV 파일 경로
        db_dir (str): 벡터 DB를 저장할 디렉토리 경로
    """
    print(f"벡터 DB 생성 시작: {input_csv} -> {db_dir}")
    
    # 파일 존재 여부 확인
    if not os.path.exists(input_csv):
        print(f"오류: 입력 파일이 존재하지 않습니다 - {input_csv}")
        return
    
    # 작업 시작 시간 기록
    total_start_time = time.time()
    
    # CSV 파일 로드
    df = load_csv_data(input_csv)
    if df.empty:
        print("처리할 데이터가 없습니다.")
        return
    
    # Document 객체 준비
    documents = prepare_documents(df)
    if not documents:
        print("변환할 문서가 없습니다.")
        return
    
    # 벡터 DB 생성
    vectordb = create_vector_db(documents, db_dir)
    
    # 벡터 DB 정보 출력
    doc_count = vectordb._collection.count()
    print(f"벡터 DB에 저장된 문서 수: {doc_count}")
    
    # 작업 완료 시간 계산
    total_elapsed_time = time.time() - total_start_time
    print(f"전체 처리 시간: {total_elapsed_time:.2f}초")
    print(f"평균 처리 시간: {total_elapsed_time/doc_count:.4f}초/문서")
    print("벡터 DB 생성 완료!")

def test_vector_search(db_dir: str, query: str, k: int = 5) -> None:
    """벡터 DB 검색 테스트 함수
    
    Args:
        db_dir (str): 벡터 DB가 저장된 디렉토리 경로
        query (str): 검색 쿼리
        k (int, optional): 검색 결과 수. 기본값은 5.
    """
    # 벡터 DB 로드
    vectordb = load_vector_db(db_dir)
    
    print(f"검색 시작: '{query}'")
    start_time = time.time()
    
    # 검색 실행
    results = vectordb.similarity_search_with_score(query, k=k)
    
    # 검색 시간 계산
    search_time = time.time() - start_time
    
    # 결과 출력
    print(f"검색 완료 ({search_time:.4f}초)")
    print(f"검색 결과 ({len(results)}개):")
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- 결과 {i+1} (유사도: {1-score:.4f}) ---")
        print(f"문장: {doc.page_content}")
        print(f"제목: {doc.metadata.get('title', '')}")
        print(f"소제목: {doc.metadata.get('subtitle', '')}")
        print(f"출처: {doc.metadata.get('source', '')}, 페이지 {doc.metadata.get('page', '')}")

if __name__ == "__main__":
    # 파일 경로 설정
    input_csv = "./code/SangGyeom/data/processed_pdf_data2.csv"
    db_dir = "./code/SangGyeom/db/vector_db"
    
    # 벡터 DB가 없는 경우에만 생성
    if not os.path.exists(db_dir):
        print("벡터 DB 디렉토리가 없습니다. 새로 생성합니다.")
        process_data_to_vector_db(input_csv, db_dir)
    else:
        print(f"벡터 DB가 이미 존재합니다: {db_dir}")
    
    # 검색 테스트
    test_vector_search(db_dir, "F.C.M 교량공사 안전보건 관련 규정은 무엇인가요?") 