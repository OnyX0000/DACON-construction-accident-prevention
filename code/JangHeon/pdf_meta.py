import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

# 1. GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. SBERT 모델 로딩
model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)

# 3. 메타데이터 기준값 로딩
with open("data/metadata_categories.json", "r", encoding="utf-8") as f:
    metadata_dict = json.load(f)

# 4. 문단 추출 함수 (LangChain 문서 로더 사용)
def extract_pdf_paragraphs(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()  # returns List[Document]

# 5. 문단 배치 임베딩 + 메타데이터 태깅 함수
def tag_batch_paragraphs(paragraphs, model, metadata_dict, threshold=0.3):
    embeddings = model.encode(paragraphs, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    results = []

    for para_emb in embeddings:
        tags = {}
        for key, candidates in metadata_dict.items():
            if not candidates:
                continue
            candidate_embs = model.encode(candidates, convert_to_tensor=True)
            scores = util.cos_sim(para_emb, candidate_embs)[0]
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()
            if best_score > threshold:
                tags[key] = candidates[best_idx]
        results.append(tags)
    return results

# 6. 전체 PDF 문서 처리 및 태깅
pdf_folder = "data/pdf"  # PDF 파일 경로
results = []
chroma_docs = []

for filename in tqdm(os.listdir(pdf_folder)):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        print(f"Processing: {filename}")
        documents = extract_pdf_paragraphs(file_path)

        paragraphs = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        tagged_list = tag_batch_paragraphs(paragraphs, model, metadata_dict)

        for para, meta, tags in zip(paragraphs, metadatas, tagged_list):
            # CSV 저장용
            results.append({
                "pdf_file": filename,
                "paragraph": para,
                "tags": tags,
                "metadata": meta
            })

            # Chroma 저장용 문서 생성
            combined_meta = meta or {}
            combined_meta.update(tags)
            combined_meta["pdf_file"] = filename

            chroma_docs.append(Document(
                page_content=para,
                metadata=combined_meta
            ))

# 7. 결과 CSV 저장
df_results = pd.DataFrame(results)
df_results.to_csv("tagged_pdf_paragraphs.csv", index=False)
print("완료: tagged_pdf_paragraphs.csv 저장됨")

# 8. Chroma DB 저장
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
chroma_path = "data/chroma_construction_db_v0"

vectorstore = Chroma.from_documents(
    documents=chroma_docs,
    embedding=embedding_model,
    persist_directory=chroma_path
)
vectorstore.persist()
print(f"완료: Chroma DB 저장됨 → {chroma_path}")