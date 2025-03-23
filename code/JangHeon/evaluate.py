import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# ✅ 파일 경로
submission_path = "submission/final_submission_with_embedding.csv"
ground_truth_path = "data/cleaned_construction_accidents.csv"
output_path = "submission/final_submission_with_scores.csv"

# ✅ 데이터 로딩
submission_df = pd.read_csv(submission_path)
gt_df = pd.read_csv(ground_truth_path)

# ✅ 평가 기준 모델 로딩
model = SentenceTransformer("jhgan/ko-sbert-sts")

# ✅ Ground Truth 딕셔너리 생성
gt_map = gt_df.set_index("사고원인")["재발방지대책 및 향후조치계획"].dropna().to_dict()

# ✅ Jaccard 유사도 함수
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

# ✅ 평가 루프
cosine_scores = []
jaccard_scores = []
references = []

for _, row in submission_df.iterrows():
    question = str(row["사고원인"])
    generated = str(row["answer"]).strip()
    reference = str(gt_map.get(question, "")).strip()
    references.append(reference)

    # Cosine 유사도
    if reference and not generated.startswith("[ERROR]"):
        emb1 = model.encode(generated, convert_to_tensor=True)
        emb2 = model.encode(reference, convert_to_tensor=True)
        cosine = util.cos_sim(emb1, emb2).item()
    else:
        cosine = None

    # Jaccard 유사도
    jaccard = jaccard_similarity(generated, reference) if reference else None

    cosine_scores.append(cosine)
    jaccard_scores.append(jaccard)

# ✅ 결과 병합 및 저장
submission_df["reference"] = references
submission_df["cosine_similarity"] = cosine_scores
submission_df["jaccard_similarity"] = jaccard_scores

os.makedirs("submission", exist_ok=True)
submission_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 평가 완료! → {output_path}")