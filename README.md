DACON_-Construction_accident_prevention
---

## 🚀 Team MDR 팀원 소개

<p align="center">
  <a href="https://github.com/vhf1030">
    <img src="https://github.com/vhf1030.png" width="120" height="120" style="border-radius:50%;" alt="vhf1030">
  </a>
  <a href="https://github.com/OnyX0000">
    <img src="https://github.com/OnyX0000.png" width="120" height="120" style="border-radius:50%;" alt="OnyX0000">
  </a>
  <a href="https://github.com/Jacod97">
    <img src="https://github.com/Jacod97.png" width="120" height="120" style="border-radius:50%;" alt="Jacod97">
  </a>
  <a href="https://github.com/LJH0963">
    <img src="https://github.com/LJH0963.png" width="120" height="120" style="border-radius:50%;" alt="LJH0963">
  </a>
  <a href="https://github.com/sokcho-kim">
    <img src="https://github.com/sokcho-kim.png" width="120" height="120" style="border-radius:50%;" alt="sokcho-kim">
  </a>
</p>

<p align="center">
  <b>👨‍💻 Team_MDR - We build the future! 🚀</b>
  <a href = "https://www.notion.so/Team_MDR-19f48fd03c228085be6cfd03c73ac223?pvs=4"> Team_MDR_notion
  </a>
</p>



### git clone
```$ git clone https://github.com/Jacod97/DACON_-Construction_accident_prevention.git```

<!-- ### 대용량 파일 관리 -->
<!-- ```$ git lfs install``` -->

### 자주쓰는 git 명령어

```
$ git log  # commit 내역 확인
```
```
$ git status  # 현재 파일 변경 상태 확인
```
```
$ git diff  # 파일 변경 내용 확인
```
```
$ git restore [파일경로]
  # 특정 파일의 수정 내용을 마지막 커밋으로 되돌리기
  # ex. git restore notebooks/eda_ikh.ipynb
  # (주의) 수정한 내용이 사라집니다
```
```
$ git branch
$ git checkout
$ git merge
```

### 서버에서 작업할 때
```
$ git -c user.name="YourName" -c user.email="YourEmail@example.com" add .  
$ git -c user.name="YourName" -c user.email="YourEmail@example.com" commit -m "커밋 메시지"  
$ git -c user.name="YourName" -c user.email="YourEmail@example.com" push
```

## 공모전 링크

#### 건설공사 사고 예방 및 대응책 생성: 한솔데코 시즌3 AI 경진대회
https://dacon.io/competitions/official/236455/overview/description

---

## 1. 개요
본 시스템은 PDF 및 CSV 데이터를 혼합하여 검색하고, LLM을 이용하여 질의응답을 수행하는 구조로 되어 있습니다. CSV 데이터는 질의응답 형식으로 변환되어 저장되며, PDF 데이터는 문단 단위로 분할 후 검색에 활용됩니다. 벡터 검색을 위해 FAISS를 사용하고, ChromaDB에 데이터를 저장하여 빠르고 효과적인 검색을 수행합니다.

## 2. 시스템 구성
### 2.1 데이터 로드 및 전처리
#### (1) PDF 데이터
- `PyMuPDFLoader`를 이용하여 PDF 문서를 문단 단위로 나누어 로드합니다.

#### (2) CSV 데이터
- 특정 컬럼을 활용하여 질의응답 형식으로 변환합니다.
```python
"question": (
    f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
    f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
    f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
    f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
    f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
)
```

### 2.2 데이터 임베딩 및 저장
- `jhgan/ko-sbert-nli` 모델을 이용하여 **텍스트 임베딩을 생성**합니다.
- PDF 데이터와 CSV 데이터를 **ChromaDB**에 저장합니다.

### 2.3 문서 검색 및 검색 전략
- `ensemble_retriever`를 활용하여 PDF 데이터와 CSV 데이터를 혼합 검색합니다.  
  - 검색 비율: **PDF(70%) : CSV(30%)**

### 2.4 질의응답 시스템
- **LLM 모델:** `ollama` 서버에서 **gemma-3b:27b** 모델을 로드하여 답변을 생성합니다.

## 3. 코드 주요 내용
### 3.1 라이브러리 및 환경 설정
```python
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
```
- `CUDA_VISIBLE_DEVICES = "1"`을 설정하여 특정 GPU만 사용하도록 제한합니다.

### 3.2 데이터 로드 및 전처리
```python
train = pd.read_csv('../../data/train.csv', encoding='utf-8-sig')
test = pd.read_csv('../../data/test.csv', encoding='utf-8-sig')
```
- 데이터를 로드한 후, 공사, 공종, 사고 객체 등을 분류별로 나누어 저장합니다.

### 3.3 임베딩 및 벡터 저장
```python
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
db = FAISS.from_documents(combined_training_data, embeddings)
```

### 3.4 문서 검색 및 질의응답 시스템
```python
retriever = db.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(model=model, tokenizer=tokenizer),
    retriever=retriever
)
```

### 3.5 Ensemble Retriever 설정
```python
from langchain.retrievers import EnsembleRetriever
combined_retriever = EnsembleRetriever(
    retrievers=[csv_retriever, pdf_retriever],
    weights=[0.3, 0.7]
)
```

### 3.6 질의응답 실행
```python
question = "이 공사의 재발 방지 대책은 무엇인가요?"
response = qa_chain.run(question)
print(response)
```

## 4. 결론
- **PDF 및 CSV 데이터를 혼합하여 검색하는 문서 검색 시스템을 구축**합니다.
- **CSV 데이터는 질의응답 형식으로 변환**하고, **PDF 데이터는 문단 단위로 처리**하여 ChromaDB에 저장합니다.
- **FAISS를 활용한 벡터 검색**과 **Ensemble Retriever(3:7 비율)**를 적용하여 효과적인 검색을 수행합니다.
- **NCSOFT의 Llama-VARCO-8B-Instruct 모델을 사용하여 질의응답을 수행**합니다.
- `ollama`를 통해 **Gemma-3b:27b 모델을 활용**할 수도 있습니다.

---

