import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer, util
import json

# PDF → Document 불러오기
def load_documents_from_folder(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# 문단 분할 (optional: 문단이 너무 길 때)
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)

# 메타데이터 태깅 함수 (선택적, 원하면 통합 가능)
with open("data/metadata_categories.json", "r", encoding="utf-8") as f:
    metadata_dict = json.load(f)

sbert_model = SentenceTransformer("jhgan/ko-sbert-nli")

def tag_metadata(text, model, metadata_dict, threshold=0.3):
    tags = {}
    para_emb = model.encode(text, convert_to_tensor=True)
    for key, candidates in metadata_dict.items():
        if not candidates:
            continue
        candidate_embs = model.encode(candidates, convert_to_tensor=True)
        scores = util.cos_sim(para_emb, candidate_embs)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        if best_score > threshold:
            tags[key] = candidates[best_idx]
    return tags

# 임베딩 모델 로딩 (HuggingFace Embeddings with ko-sbert)
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# Chroma에 저장
def save_to_chroma(docs, persist_directory="data/chroma_construction_db"):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Chroma DB 저장 완료: {persist_directory}")

# 전체 파이프라인 실행
pdf_folder = "data/pdf"
raw_docs = load_documents_from_folder(pdf_folder)
print(f"원본 문서 수: {len(raw_docs)}")

split_docs = split_documents(raw_docs)
print(f"분할된 문단 수: {len(split_docs)}")

# 각 문단에 자동 메타데이터 태깅 추가
for doc in split_docs:
    tags = tag_metadata(doc.page_content, sbert_model, metadata_dict)
    doc.metadata.update(tags)

# 저장
save_to_chroma(split_docs)