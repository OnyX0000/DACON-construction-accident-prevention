#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import importlib.util
import csv
from datetime import datetime

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
calculate_cosine_similarity = rag_module.calculate_cosine_similarity

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

def format_results(original, generated, similarity, prompt=None):
    """결과를 출력하는 함수"""
    print("\n" + "=" * 100)
    print(f"코사인 유사도: {similarity:.4f}")
    
    if prompt:
        print("-" * 100)
        print(f"사용된 프롬프트:")
        print(prompt)
    
    print("-" * 100)
    print(f"원본 답변 (CSV): {original}")
    print("-" * 100)
    print(f"생성된 답변: {generated}")
    print("=" * 100)

def get_relevant_context(vectordb, query, k=5):
    """쿼리와 관련된 문맥을 검색하는 함수"""
    docs = vectordb.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def save_results_to_csv(results, file_path):
    """결과를 CSV 파일로 저장하는 함수"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['원본 답변', '생성된 답변', '코사인 유사도'])
        
        for result in results:
            writer.writerow([
                result['original'],
                result['generated'],
                result['similarity']
            ])
    
    print(f"결과가 CSV 파일에 저장되었습니다: {file_path}")
    return file_path

def compare_csv_answers(csv_file_path):
    """CSV 파일의 원본 답변과 생성된 답변을 비교하여 출력하는 함수"""
    if not os.path.exists(csv_file_path):
        print(f"CSV 파일이 존재하지 않습니다: {csv_file_path}")
        return
    
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        
        print("\n" + "=" * 100)
        print(f"CSV 파일의 결과 비교 ({csv_file_path}):")
        print("=" * 100)
        
        for i, row in enumerate(df.itertuples(), 1):
            original = row[1]  # 첫 번째 열: 원본 답변
            generated = row[2]  # 두 번째 열: 생성된 답변
            similarity = row[3]  # 세 번째 열: 코사인 유사도
            
            print(f"\n[결과 {i}]")
            print(f"코사인 유사도: {similarity:.4f}")
            print("-" * 80)
            print(f"원본 답변 (CSV): {original}")
            print("-" * 80)
            print(f"생성된 답변: {generated}")
            print("-" * 80)
        
        # 평균 코사인 유사도 계산 및 출력
        avg_similarity = df.iloc[:, 2].mean()
        print("\n" + "=" * 50)
        print(f"총 {len(df)}개 결과의 평균 코사인 유사도: {avg_similarity:.4f}")
        print("=" * 50)
    
    except Exception as e:
        print(f"CSV 파일 분석 중 오류 발생: {str(e)}")

def initialize_gemma3_model(model_name="gemma3:27b"):
    """Ollama를 통해 Gemma3 모델을 초기화하는 함수"""
    try:
        # hsdeco.ipynb 참고하여 Ollama 모델 초기화
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
    
    return {
        "question": question,
        "answer": answer,
        "source_docs": source_docs,
        "elapsed_time": elapsed_time
    }

def run_rag_test(train_csv, vector_db_dir, model_name, test_size=10):
    """RAG 테스트를 실행하는 함수"""
    print(f"RAG 테스트 시작 (테스트 크기: {test_size})")
    
    # 랭체인 프롬프트 템플릿 정보 출력
    print_prompt_info()
    
    # 1. 학습 데이터 로드
    try:
        train_df = pd.read_csv(train_csv, encoding='utf-8-sig')
        print(f"학습 데이터 로드 완료: {train_df.shape[0]} 행, {train_df.shape[1]} 열")
    except Exception as e:
        print(f"학습 데이터 로드 오류: {str(e)}")
        return
    
    # 테스트 크기 조정
    test_size = min(test_size, len(train_df))
    test_df = train_df.head(test_size).copy()
    
    # 2. 벡터 DB 로드
    vectordb = load_vector_db(vector_db_dir)
    if vectordb is None:
        print(f"벡터 DB 로드 실패: {vector_db_dir}")
        return
    
    # 3. Gemma3 LLM 모델 초기화
    llm = initialize_gemma3_model(model_name)
    if llm is None:
        print("Gemma3 모델 초기화 실패")
        return
    
    # 4. RAG 체인 생성
    chain = create_rag_chain(vectordb, llm)
    
    # 5. 각 행에 대해 쿼리 및 결과 생성
    results = []
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RAG 테스트"):
        # 사고 원인만 쿼리로 사용
        query = row["사고원인"]
        
        # 원본 답변 가져오기
        original_answer = row["재발방지대책 및 향후조치계획"]
        
        # 관련 문맥 가져오기
        context = get_relevant_context(vectordb, query)
        
        # 최종 프롬프트 생성
        prompt_template = get_prompt_template()
        final_prompt = format_final_prompt(prompt_template, context, query)
        
        # RAG로 답변 생성
        result = generate_answer(chain, query)
        generated_answer = result["answer"]
        
        # 코사인 유사도 계산
        similarity = calculate_cosine_similarity(original_answer, generated_answer)
        
        # 결과 저장
        results.append({
            "original": original_answer,
            "generated": generated_answer,
            "similarity": similarity,
            "prompt": final_prompt
        })
        
        # 결과 출력
        format_results(original_answer, generated_answer, similarity, final_prompt)
    
    # 6. 결과를 CSV 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = f"./code/SangGyeom/results/rag_test_gemma3_results_{timestamp}.csv"
    saved_csv_path = save_results_to_csv(results, csv_file_path)
    
    # 7. 저장된 CSV 파일의 결과 비교 출력
    compare_csv_answers(saved_csv_path)
    
    # 8. 요약 통계 출력
    similarities = [r["similarity"] for r in results]
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    print("\n" + "=" * 50)
    print(f"테스트 요약 (테스트 크기: {test_size})")
    print(f"평균 코사인 유사도: {avg_similarity:.4f}")
    print(f"최소 코사인 유사도: {min_similarity:.4f}")
    print(f"최대 코사인 유사도: {max_similarity:.4f}")
    print(f"결과 CSV 파일: {saved_csv_path}")
    print("=" * 50)

if __name__ == "__main__":
    # 경로 설정
    train_csv = "./code/SangGyeom/data/train.csv"
    vector_db_dir = "./code/SangGyeom/db/vector_db" 
    
    # 벡터 DB 디렉토리가 존재하는지 확인
    if not os.path.exists(vector_db_dir):
        print(f"경고: 벡터 DB 디렉토리 '{vector_db_dir}'가 존재하지 않습니다.")
        print("벡터 DB 디렉토리가 없으면 04_RAG.py를 먼저 실행해야 합니다.")
        print("프로그램을 종료합니다.")
        sys.exit(1)
    
    # Gemma3 모델 설정 (hsdeco.ipynb 참고)
    model_name = "gemma3:27b"
    
    # 테스트 실행
    run_rag_test(
        train_csv=train_csv,
        vector_db_dir=vector_db_dir,
        model_name=model_name,
        test_size=10
    )
