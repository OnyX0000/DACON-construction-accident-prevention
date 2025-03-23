import os
import glob
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import torch
from functools import partial
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def extract_metadata_from_filename(filename):
    """파일명에서 메타데이터 추출"""
    # 파일명에서 확장자 제거
    filename = os.path.splitext(filename)[0]
    
    # 공사 종류 분류
    construction_types = {
        '건축': ['건축물', '건설공사', '건설현장', '건설기계', '건설업체'],
        '토목': ['교량', '터널', '도로', '철도', '항만', '하천'],
        '조경': ['조경', '수목', '식재'],
        '설비': ['설비', '플랜트', '시스템', '기계'],
        '기타': []
    }
    
    # 공종 분류
    work_types = {
        '기초공사': ['기초', '말뚝', '지하', '굴착'],
        '구조공사': ['철골', '콘크리트', '거푸집', '동바리'],
        '설비공사': ['설비', '배관', '전기', '용접'],
        '마감공사': ['미장', '타일', '도배', '페인트'],
        '기타': []
    }
    
    # 사고 유형 분류
    accident_types = {
        '추락': ['추락', '고소', '비계', '발판'],
        '낙하': ['낙하', '물체', '중량물'],
        '굴착': ['굴착', '터널', '지하'],
        '감전': ['전기', '감전'],
        '기타': []
    }
    
    # 사고 객체 분류
    accident_objects = {
        '건설기계': ['크레인', '펌프', '굴착기', '타워크레인'],
        '건설자재': ['철근', '콘크리트', '거푸집', '동바리'],
        '설비': ['전기설비', '배관', '용접기'],
        '기타': []
    }
    
    # 메타데이터 초기화
    metadata = {
        'construction_type': '기타',  # 공사종류
        'work_type': '기타',         # 공종
        'accident_type': '기타',     # 사고유형
        'accident_object': '기타',   # 사고객체
        'specific_method': '',       # 특정 공법
        'equipment': '',             # 사용 장비
        'location': '',              # 작업 위치
        'work_process': '',          # 작업프로세스
        'accident_cause': '',        # 사고원인
        'prevention_measure': '',    # 재발방지대책
        'future_plan': ''            # 향후조치계획
    }
    
    # 공사 종류 분류
    for const_type, keywords in construction_types.items():
        if any(keyword in filename for keyword in keywords):
            metadata['construction_type'] = const_type
            break
    
    # 공종 분류
    for work_type, keywords in work_types.items():
        if any(keyword in filename for keyword in keywords):
            metadata['work_type'] = work_type
            break
    
    # 사고 유형 분류
    for acc_type, keywords in accident_types.items():
        if any(keyword in filename for keyword in keywords):
            metadata['accident_type'] = acc_type
            break
    
    # 사고 객체 분류
    for acc_obj, keywords in accident_objects.items():
        if any(keyword in filename for keyword in keywords):
            metadata['accident_object'] = acc_obj
            break
    
    # 특정 공법 추출
    method_patterns = [
        r'\((.*?)\)',  # 괄호 안의 내용
        r'공법',       # '공법' 앞의 내용
        r'방법'        # '방법' 앞의 내용
    ]
    
    for pattern in method_patterns:
        match = re.search(pattern, filename)
        if match:
            metadata['specific_method'] = match.group(1)
            break
    
    # 장비 추출
    equipment_keywords = ['크레인', '비계', '동바리', '거푸집', '발판', '비계']
    for keyword in equipment_keywords:
        if keyword in filename:
            metadata['equipment'] = keyword
            break
    
    return metadata

def extract_sections(text):
    """텍스트를 섹션별로 분리"""
    sections = []
    current_section = ""
    
    for line in text.split('\n'):
        # 섹션 시작 패턴 (숫자로 시작하는 줄)
        if re.match(r'^\d+\.\s+', line):
            if current_section:
                sections.append(current_section.strip())
            current_section = line
        else:
            current_section += "\n" + line
    
    if current_section:
        sections.append(current_section.strip())
    
    return sections

def create_vector_database(documents: List[Dict[str, Any]], model_name='jhgan/ko-sbert-sts', batch_size=32):
    """벡터 데이터베이스 생성 (배치 처리 및 GPU 가속 적용)"""
    print("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 문서 분할 및 벡터 저장소 생성
    print("문서 분할 및 벡터 저장소 생성 중...")
    texts = []
    metadatas = []
    
    # 배치 처리로 문서 분할
    for i in tqdm(range(0, len(documents), batch_size)):
        batch_docs = documents[i:i + batch_size]
        batch_texts = []
        batch_metadatas = []
        
        for doc in batch_docs:
            splits = text_splitter.split_text(doc["content"])
            batch_texts.extend(splits)
            batch_metadatas.extend([doc["metadata"]] * len(splits))
        
        texts.extend(batch_texts)
        metadatas.extend(batch_metadatas)
    
    # FAISS 벡터 저장소 생성 (배치 처리)
    print("벡터 저장소 생성 중...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    return vectorstore

def process_document_batch(documents: List[Dict[str, Any]], batch_size: int = 32):
    """문서 배치 처리"""
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_results = []
        for doc in batch:
            try:
                with open(doc['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    metadata = extract_metadata_from_filename(doc['filename'])
                    metadata['filename'] = doc['filename']
                    
                    document = {
                        "metadata": metadata,
                        "content": content,
                        "sections": extract_sections(content)
                    }
                    batch_results.append(document)
            except Exception as e:
                print(f"Error processing {doc['filename']}: {str(e)}")
        results.extend(batch_results)
    return results

def load_and_process_documents(txt_path: str, batch_size: int = 32, max_workers: int = 4):
    """텍스트 파일들을 병렬로 로드하고 처리"""
    txt_files = glob.glob(os.path.join(txt_path, '*.txt'))
    documents = [{'file_path': f, 'filename': os.path.basename(f)} for f in txt_files]
    
    # 병렬 처리로 문서 로드
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            futures.append(executor.submit(process_document_batch, batch, batch_size))
        
        processed_documents = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="문서 처리 중"):
            processed_documents.extend(future.result())
    
    return processed_documents

def save_vector_database(vectorstore, save_path):
    """벡터 데이터베이스 저장"""
    print("벡터 데이터베이스 저장 중...")
    vectorstore.save_local(save_path)

def load_vector_database(load_path, model_name='jhgan/ko-sbert-sts'):
    """벡터 데이터베이스 로드"""
    print("벡터 데이터베이스 로드 중...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(load_path, embeddings)
    return vectorstore

def search_similar_sections(query, vectorstore, k=5):
    """유사한 섹션 검색"""
    # 유사도 검색
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # 결과 반환
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            'section': doc.page_content,
            'similarity': 1 - score,  # 거리를 유사도로 변환
            'metadata': doc.metadata
        })
    
    return formatted_results

def create_embeddings_batch(texts: List[str], model_name: str = "jhgan/ko-sbert-sts", batch_size: int = 32):
    """배치 처리로 임베딩 생성 (GPU 가속 적용)"""
    embedding = SentenceTransformer(model_name)
    embedding.to(device)  # GPU로 모델 이동
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성 중"):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding.encode(
            batch,
            show_progress_bar=False,
            device=device,
            convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def process_rag_batch(rag_chain, questions: List[str], batch_size: int = 8):
    """RAG 처리 배치 (GPU 가속 적용)"""
    results = []
    for i in tqdm(range(0, len(questions), batch_size), desc="RAG 처리 중"):
        batch_questions = questions[i:i + batch_size]
        batch_results = []
        for question in batch_questions:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = rag_chain.invoke({"question": question})
            batch_results.append(result)
        results.extend(batch_results)
    return results

def load_test_data():
    """테스트 데이터 로드"""
    print("테스트 데이터 로드 중...")
    test_data = pd.read_csv('./test.csv', encoding='utf-8-sig')
    return test_data

def main():
    # GPU 메모리 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # 테스트 데이터 로드
    combined_test_data = load_test_data()
    
    # 경로 설정
    txt_path = "/home/wanted-1/potenup-workspace/Project/dacon/DACON-construction-accident-prevention/data/text_output"
    vector_db_path = "./db"
    
    # 벡터 DB 디렉토리 생성
    os.makedirs(vector_db_path, exist_ok=True)
    
    # 문서 로드 및 처리 (병렬 처리 적용)
    documents = load_and_process_documents(txt_path, batch_size=32, max_workers=4)
    print(f"총 {len(documents)}개의 문서가 로드되었습니다.")
    
    # 벡터 데이터베이스 생성 (배치 처리 및 GPU 가속 적용)
    vectorstore = create_vector_database(documents, batch_size=32)
    
    # 벡터 데이터베이스 저장
    save_vector_database(vectorstore, vector_db_path)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOllama(model='gemma3:27b', temperature=0.0)

    template = """
    {context}

    ### 질문:
    {question}

    ### 지침: 당신은 건설 안전 전문가입니다.
    질문에 대한 답변을 한 문장으로 작성해 주세요..
    - 최대 64토큰 이내로 간결하게 작성해 주세요.
    - 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
    - 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.
    - "합니다", "했다" 등의 동사를 사용하지 마세요.
    - **"수립했습니다" 대신 "수립."과 같이 문장을 끝맺으세요.**
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_retrieved_docs(docs):
        """retriever가 반환한 문서를 문자열로 변환"""
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {
            'context': lambda inputs: format_retrieved_docs(retriever.get_relevant_documents(inputs['question'])),
            'question': itemgetter("question")
        }
        | prompt 
        | llm
        | StrOutputParser()
    )
    
    # 테스트 실행 및 결과 저장
    print("테스트 실행 시작... 총 테스트 샘플 수:", len(combined_test_data))
    
    # 배치 처리로 RAG 실행 (GPU 가속 적용)
    questions = combined_test_data['question'].tolist()
    test_results = process_rag_batch(rag_chain, questions, batch_size=8)
    
    print("\n테스트 실행 완료! 총 결과 수:", len(test_results))
    
    # 배치 처리로 임베딩 생성 (GPU 가속 적용)
    embedding_model_name = "jhgan/ko-sbert-sts"
    pred_embeddings = create_embeddings_batch(test_results, batch_size=32)
    print(pred_embeddings.shape)  # (샘플 개수, 768)

    # 결과 저장
    submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    submission.iloc[:,1] = test_results
    submission.iloc[:,2:] = pred_embeddings
    
    # 최종 결과를 CSV로 저장
    submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return test_results

if __name__ == "__main__":
    main() 