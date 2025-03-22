import os
import re
import time
import argparse
from typing import List, Dict, Any, Tuple

# 최신 LangChain API 사용
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

def load_vector_db(db_dir: str) -> Chroma:
    """벡터 DB를 로드하는 함수
    
    Args:
        db_dir (str): 벡터 DB 디렉토리 경로
        
    Returns:
        Chroma: 로드된 벡터 DB
    """
    # 임베딩 모델 설정 - Baseline 코드와 일치시킴
    embedding_model_name = "jhgan/ko-sbert-nli"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # 벡터 DB 로드
    try:
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embedding
        )
        print(f"벡터 DB 로드 완료: {db_dir}")
        print(f"총 문서 수: {vectordb._collection.count()}")
        return vectordb
    except Exception as e:
        print(f"벡터 DB 로드 오류: {str(e)}")
        return None

def advanced_query(vectordb: Chroma, query: str, k: int = 5) -> Dict[Tuple[str, str], List[Document]]:
    """고급 검색 쿼리 실행 함수
    
    Args:
        vectordb (Chroma): 검색할 벡터 DB
        query (str): 검색 쿼리
        k (int): 검색 결과 수
        
    Returns:
        Dict[Tuple[str, str], List[Document]]: 제목과 소제목으로 그룹화된 결과
    """
    print(f"검색 시작: '{query}'")
    
    # 1. 기본 검색 수행
    start_time = time.time()
    results = vectordb.similarity_search_with_score(query, k=k)
    query_time = time.time() - start_time
    
    print(f"기본 검색 완료 ({query_time:.4f}초), 결과 {len(results)}개")
    
    if not results:
        print("검색 결과가 없습니다.")
        return {}
    
    # 2. 가장 유사도가 높은 문서의 제목 확인
    top_doc, top_score = results[0]
    top_title = top_doc.metadata.get('title', '')
    
    print(f"가장 유사도가 높은 문서 제목: '{top_title}', 유사도: {1-top_score:.4f}")
    
    # 3. 제목 필터링 및 "목적" 제외
    filtered_results = []
    
    for doc, score in results:
        doc_title = doc.metadata.get('title', '')
        doc_subtitle = doc.metadata.get('subtitle', '')
        
        # 동일한 제목만 포함
        if doc_title != top_title:
            continue
        
        # "목적" 패턴이 있는 소제목 제외 - 정확한 단어 매칭으로 수정
        if re.search(r'목\s*적', doc_subtitle):
            print(f"소제목 제외: '{doc_subtitle}' - 목적 관련 패턴 검출")
            continue
        
        filtered_results.append((doc, score))
    
    if not filtered_results:
        print("필터링 후 남은 결과가 없습니다.")
        return {}
    
    # 4. 가장 유사도가 높은 소제목 확인
    top_filtered_doc, _ = filtered_results[0]
    top_subtitle = top_filtered_doc.metadata.get('subtitle', '')
    
    print(f"선택된 소제목: '{top_subtitle}'")
    
    # 5. 모든 문서 가져오기
    all_docs = vectordb._collection.get()
    docs = all_docs['documents']
    metadatas = all_docs['metadatas']
    
    # 6. 제목과 소제목이 일치하는 모든 문서 수집
    matching_docs = []
    
    for i, metadata in enumerate(metadatas):
        if (metadata.get('title') == top_title and 
            metadata.get('subtitle') == top_subtitle):
            doc = Document(
                page_content=docs[i],
                metadata=metadata
            )
            matching_docs.append(doc)
    
    # 7. 결과 그룹화 - (제목, 소제목) 튜플을 키로 사용
    group_key = (top_title, top_subtitle)
    grouped_results = {group_key: matching_docs}
    
    print(f"매칭 문서 수: {len(matching_docs)}")
    
    return grouped_results

def display_grouped_results(grouped_results: Dict[Tuple[str, str], List[Document]]) -> None:
    """그룹화된 검색 결과 출력 함수
    
    Args:
        grouped_results (Dict[Tuple[str, str], List[Document]]): 그룹화된 검색 결과
    """
    if not grouped_results:
        print("표시할 결과가 없습니다.")
        return
    
    for i, (group_key, docs) in enumerate(grouped_results.items()):
        title, subtitle = group_key
        
        print("\n" + "=" * 80)
        print(f"결과 그룹 {i+1}")
        print("=" * 80)
        print(f"제목: {title}")
        print(f"소제목: {subtitle}")
        print(f"문서 수: {len(docs)}")
        
        # 문서 정렬 (페이지 및 라인 번호 기준)
        sorted_docs = sorted(docs, key=lambda d: (
            int(d.metadata.get('page', 0)), 
            int(d.metadata.get('line_num', 0))
        ))
        
        # 모든 문장 합치기
        combined_text = " ".join([doc.page_content for doc in sorted_docs])
        
        print("\n--- 통합 내용 ---")
        print(combined_text)
        
        # 첫 문서의 출처 정보
        if sorted_docs:
            first_doc = sorted_docs[0]
            source = first_doc.metadata.get('source', '')
            page = first_doc.metadata.get('page', '')
            print(f"\n출처: {source}")
            print(f"시작 페이지: {page}")

def run_query(db_dir: str, query: str) -> None:
    """쿼리 실행 함수
    
    Args:
        db_dir (str): 벡터 DB 디렉토리 경로
        query (str): 검색 쿼리
    """
    # 벡터 DB 로드
    vectordb = load_vector_db(db_dir)
    if not vectordb:
        return
    
    # 고급 검색 실행
    grouped_results = advanced_query(vectordb, query)
    
    # 결과 출력
    display_grouped_results(grouped_results)

if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='벡터 DB 고급 검색')
    parser.add_argument('--db_dir', type=str, default='./code/SangGyeom/db/vector_db',
                        help='벡터 DB 디렉토리 경로')
    parser.add_argument('--query', type=str, default='고소작업 중 추락 위험이 있음에도 불구하고, 안전난간대, 안전고리 착용 등 안전장치가 미흡하였음.',
                        help='검색 쿼리')
    args = parser.parse_args()
    
    # 쿼리 실행
    run_query(args.db_dir, args.query) 
    
# python ./code/SangGyeom/03_1_query_test.py --query "3층 슬라브 작업시 이동중  미끄러짐"