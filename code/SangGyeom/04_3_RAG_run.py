#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import importlib.util
from datetime import datetime
from sentence_transformers import SentenceTransformer

# 최신 LangChain API 사용
from langchain.schema.document import Document
from langchain.llms import Ollama

# 04_RAG.py 모듈 동적 로드
module_path = os.path.join(os.path.dirname(__file__), "04_RAG.py")
spec = importlib.util.spec_from_file_location("rag_module", module_path)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)

# 필요한 함수 가져오기
load_vector_db = rag_module.load_vector_db

def get_prompt_template():
    """04_RAG.py에서 사용되는 프롬프트 템플릿을 가져오는 함수"""
    prompt_template = """
    ### 지침: 당신은 건설 안전 전문가입니다.
    답변은 반드시 핵심 용어와 주요 문구만을 사용하여 작성해야 합니다.
    서론, 배경 설명, 부연 설명은 절대 포함하지 마세요.
    답변은 간결하고 명확하게 최대 64 토큰 이내로 간결하게 작성하세요.
    질문이 구체적이지 않거나 참고 문맥의 정보가 부족한 경우 경우 다음과 같이 답변하세요:
    "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."
    
    ### 참고 문맥:
    {context}
    
    ### 질문:
    {question}

    [/INST]
    """
    return prompt_template

def format_final_prompt(prompt_template, context, question):
    """최종 프롬프트를 포맷팅하는 함수"""
    return prompt_template.format(context=context, question=question)

def print_prompt_info():
    """프롬프트 템플릿 정보를 출력하는 함수"""
    prompt_template = get_prompt_template()
    print("\n" + "=" * 50)
    print("랭체인 프롬프트 템플릿:")
    print("-" * 50)
    print(prompt_template)
    print("=" * 50)

def get_relevant_context(vectordb, query, k=5):
    """쿼리와 관련된 문맥을 검색하는 함수"""
    docs = vectordb.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def initialize_gemma3_model(model_name="gemma3:27b"):
    """Ollama를 통해 Gemma3 모델을 초기화하는 함수"""
    try:
        llm = Ollama(model=model_name, temperature=0)
        print(f"Gemma3 모델 초기화 완료: {model_name}")
        return llm
    except Exception as e:
        print(f"Gemma3 모델 초기화 오류: {str(e)}")
        return None

def create_rag_chain(vectordb, llm):
    """RAG 체인을 생성하는 함수"""
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    
    # 프롬프트 템플릿 생성
    prompt_template = get_prompt_template()
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Retriever 설정
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # RetrievalQA 체인 생성
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

def generate_answer(chain, question):
    """질문에 대한 답변을 생성하는 함수"""
    print(f"질문: {question[:100]}...")
    
    start_time = time.time()
    result = chain.invoke({"query": question})
    elapsed_time = time.time() - start_time
    
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    print(f"답변 생성 완료 ({elapsed_time:.2f}초)")
    print(f"생성된 답변: {answer}")
    print("-" * 50)
    
    return answer

def run_rag_inference():
    """RAG 추론을 실행하고 submission 파일을 생성하는 함수"""
    print("RAG 추론 시작")
    
    # 랭체인 프롬프트 템플릿 정보 출력
    print_prompt_info()
    
    # 1. 테스트 데이터 로드
    test_csv = "./code/SangGyeom/data/test.csv"
    vector_db_dir = "./code/SangGyeom/db/vector_db"
    
    try:
        test_df = pd.read_csv(test_csv, encoding='utf-8-sig')
        print(f"테스트 데이터 로드 완료: {test_df.shape[0]} 행, {test_df.shape[1]} 열")
    except Exception as e:
        print(f"테스트 데이터 로드 오류: {str(e)}")
        return
    
    # 2. 벡터 DB 로드
    vectordb = load_vector_db(vector_db_dir)
    if vectordb is None:
        print(f"벡터 DB 로드 실패: {vector_db_dir}")
        return
    
    # 3. Gemma3 LLM 모델 초기화
    model_name = "gemma3:27b"
    llm = initialize_gemma3_model(model_name)
    if llm is None:
        print("Gemma3 모델 초기화 실패")
        return
    
    # 4. RAG 체인 생성
    chain = create_rag_chain(vectordb, llm)
    
    # 5. 각 행에 대해 쿼리 및 결과 생성
    test_results = []
    
    print(f"추론 시작... 총 테스트 샘플 수: {len(test_df)}")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 추론"):
        # 50개당 한 번 진행 상황 출력
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"\n[샘플 {idx + 1}/{len(test_df)}] 진행 중...")
        
        # 사고 원인만 쿼리로 사용
        query = row["사고원인"]
        
        # 상세 정보 출력
        print("\n" + "=" * 70)
        print(f"샘플 {idx + 1} 처리 중:")
        print(f"ID: {row['ID']}")
        print(f"쿼리(사고원인): {query}")
        print("-" * 70)
        
        # 관련 문맥 가져오기
        context = get_relevant_context(vectordb, query)
        
        # 최종 프롬프트 생성
        prompt_template = get_prompt_template()
        final_prompt = format_final_prompt(prompt_template, context, query)
        
        # RAG로 답변 생성
        generated_answer = generate_answer(chain, query)
        
        # 결과 저장
        test_results.append(generated_answer)
        
        print(f"샘플 {idx + 1} 처리 완료")
        print("=" * 70)
    
    print(f"\n추론 완료! 총 결과 수: {len(test_results)}")
    
    # 6. 제출 파일 생성
    # 테스트 ID 가져오기
    test_ids = test_df['ID'].tolist()
    
    # 임베딩 생성
    embedding_model_name = "jhgan/ko-sbert-sts"
    embedding_model = SentenceTransformer(embedding_model_name)
    
    print("임베딩 생성 중...")
    pred_embeddings = embedding_model.encode(test_results)
    print(f"임베딩 생성 완료! 형태: {pred_embeddings.shape}")  # (샘플 개수, 768)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame()
    submission_df['ID'] = test_ids
    submission_df['재발방지대책'] = test_results
    
    # 임베딩을 데이터프레임에 추가
    embedding_cols = [f'embedding_{i}' for i in range(pred_embeddings.shape[1])]
    embedding_df = pd.DataFrame(pred_embeddings, columns=embedding_cols)
    
    # 최종 제출 데이터프레임 생성
    final_submission = pd.concat([submission_df, embedding_df], axis=1)
    
    # 파일 저장
    submission_path = "./code/SangGyeom/data/submission0322.csv"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    final_submission.to_csv(submission_path, index=False, encoding='utf-8-sig')
    print(f"제출 파일이 생성되었습니다: {submission_path}")

if __name__ == "__main__":
    # 벡터 DB 디렉토리가 존재하는지 확인
    vector_db_dir = "./code/SangGyeom/db/vector_db" 
    if not os.path.exists(vector_db_dir):
        print(f"경고: 벡터 DB 디렉토리 '{vector_db_dir}'가 존재하지 않습니다.")
        print("벡터 DB 디렉토리가 없으면 04_RAG.py를 먼저 실행해야 합니다.")
        print("프로그램을 종료합니다.")
        sys.exit(1)
    
    # RAG 추론 실행
    run_rag_inference()
