import os
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

# 최신 LangChain API 사용
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Hugging Face 모델 관련 라이브러리
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

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

def calculate_cosine_similarity(text1: str, text2: str, model_name: str = "jhgan/ko-sbert-nli") -> float:
    """두 텍스트 간의 코사인 유사도를 계산하는 함수"""
    model = SentenceTransformer(model_name)
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    return cos_sim

def create_embeddings(texts: List[str], model_name: str = "jhgan/ko-sbert-nli") -> np.ndarray:
    """텍스트 리스트의 임베딩을 생성하는 함수"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def initialize_rag_model(model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct") -> HuggingFacePipeline:
    """RAG에 사용할 LLM 모델을 초기화하는 함수"""
    print(f"LLM 모델 초기화 중: {model_name}")
    
    try:
        # 양자화 설정 확인
        use_quantization = True
        try:
            import bitsandbytes
        except ImportError:
            use_quantization = False
            print("bitsandbytes 패키지를 찾을 수 없어 양자화를 사용하지 않습니다.")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드 설정
        model_kwargs = {"device_map": "auto"}
        
        # 양자화 설정 추가
        if use_quantization:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs["quantization_config"] = bnb_config
            except Exception:
                use_quantization = False
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # 파이프라인 생성
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        # LLM 객체 생성
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"LLM 모델 초기화 완료: {model_name}")
        return llm
    
    except Exception as e:
        print(f"LLM 모델 초기화 오류: {str(e)}")
        return None

def create_rag_chain(vectordb: Chroma, llm: Any) -> RetrievalQA:
    """RAG 체인을 생성하는 함수"""
    # 프롬프트 템플릿 생성
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
    
    ### 답변:
    """
    
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

def generate_answer(chain: RetrievalQA, question: str) -> Dict[str, Any]:
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

if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='건설 안전 RAG 시스템')
    parser.add_argument('--train_csv', type=str, default='./code/SangGyeom/data/train.csv',
                        help='학습 데이터 CSV 경로')
    parser.add_argument('--db_dir', type=str, default='./code/SangGyeom/db/vector_db',
                        help='벡터 DB 저장 디렉토리 경로')
    args = parser.parse_args()
    
    print("벡터 DB 생성 기능이 제거되었습니다.")
    print("이 스크립트는 이제 벡터 DB를 생성하지 않습니다.")
    print("벡터 DB가 필요한 경우 별도의 스크립트를 사용하세요.") 