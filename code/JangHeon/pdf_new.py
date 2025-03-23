import os
import fitz  # PyMuPDF
from PIL import Image
import io
import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OCR 엔진 설정: EasyOCR가 설치되어 있으면 사용, 아니면 pytesseract 사용
try:
    import easyocr
    reader = easyocr.Reader(['ko'], gpu=True)
    def ocr_image(image):
        results = reader.readtext(image)
        return "\n".join([text for _, text, _ in results])
    print("[INFO] EasyOCR 사용 중 (GPU 지원)")
except ImportError:
    import pytesseract
    def ocr_image(image):
        return pytesseract.image_to_string(image, lang='kor')
    print("[INFO] EasyOCR 미설치 → pytesseract 사용 중 (CPU)")

# 분석할 키워드 목록
KEYWORDS = ["사고", "안전", "방지", "대책"]

# TF-IDF 기반 자동 분류 함수
CATEGORY_MAP = {
    "공사_대분류": {
        "건축": ["건축", "슬래브", "커튼월", "조적"],
        "토목": ["토목", "교량", "터널", "암거"],
        "전기": ["전기", "전선", "배선"],
        "설비": ["설비", "배관", "덕트"],
        "플랜트": ["플랜트", "배관설치"]
    },
    "공사_중분류": {
        "공동주택": ["아파트", "공동주택"],
        "상업시설": ["상가", "오피스텔"],
        "공공기관": ["학교", "시청", "공공기관"]
    },
    "공사_소분류": {
        "아파트": ["아파트"],
        "오피스텔": ["오피스텔"],
        "학교": ["학교"]
    },
    "공종_대분류": {
        "골조": ["거푸집", "철근", "콘크리트"],
        "마감": ["미장", "타일", "도장"],
        "토공": ["굴착", "흙막이"],
        "철근콘크리트": ["철근콘크리트", "RC 공사"]
    },
    "공종_세부": {
        "타설": ["타설"],
        "비계설치": ["비계 설치", "비계 작업"],
        "설비배관": ["배관 작업", "배관 설치"],
        "전선포설": ["전선 포설", "배선 작업"]
    },
    "사고객체_대분류": {
        "비계": ["비계"],
        "거푸집": ["거푸집"],
        "장비": ["크레인", "타워크레인", "양중기"],
        "철근": ["철근"]
    },
    "사고객체_세부": {
        "시스템비계": ["시스템비계"],
        "앵커": ["앵커"],
        "드릴": ["드릴"],
        "콘크리트": ["콘크리트"]
    },
    "장소_대분류": {
        "지상": ["지상"],
        "지하": ["지하"],
        "옥상": ["옥상"],
        "주차장": ["주차장"]
    },
    "장소_세부": {
        "계단실": ["계단실"],
        "기계실": ["기계실"],
        "맨홀": ["맨홀"],
        "출입구": ["출입구"]
    },
    "부위_대분류": {
        "팔": ["팔", "팔꿈치"],
        "다리": ["다리", "무릎", "발목"],
        "머리": ["머리"],
        "허리": ["허리"]
    },
    "부위_세부": {
        "손가락": ["손가락"],
        "발목": ["발목"],
        "무릎": ["무릎"],
        "어깨": ["어깨"]
    },
    "인적사고": {
        "부주의": ["부주의"],
        "미끄러짐": ["미끄러짐"],
        "추락": ["추락"],
        "협착": ["협착"]
    },
    "물적사고": {
        "자재파손": ["자재 파손"],
        "장비고장": ["장비 고장"],
        "낙하물": ["낙하물"]
    },
    "작업프로세스": {
        "타설": ["타설"],
        "운반": ["운반"],
        "해체": ["해체"],
        "설치": ["설치"],
        "점검": ["점검"]
    },
    "사고원인": {
        "부주의": ["부주의"],
        "안전조치 미비": ["안전조치 미비", "안전장치 없음"],
        "순서 불이행": ["작업순서 무시", "절차 미준수"]
    }
}  # 기존 CATEGORY_MAP은 생략됨, 기존 코드와 동일하게 유지

# TF-IDF 기반 가장 유사한 라벨 찾기
def tfidf_best_match(text, category_dict):
    labels = list(category_dict.keys())
    candidates = [" ".join(keywords) for keywords in category_dict.values()]
    corpus = candidates + [text]
    vectorizer = TfidfVectorizer().fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)
    sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    best_idx = sim.argmax()
    if sim[best_idx] > 0.1:  # 유사도 임계값 설정
        return labels[best_idx]
    return ""

# 텍스트 기반 자동 분류 함수
def auto_classify(text, mapping):
    result = {}
    for field, category_dict in mapping.items():
        result[field] = tfidf_best_match(text, category_dict)
    return result

# 결과 저장 폴더 생성
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 텍스트에서 키워드가 있는 문장만 추출
def extract_relevant_sentences(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if any(k in line for k in KEYWORDS)]

# PDF에서 텍스트 추출 (OCR 포함)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    result = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        if not text.strip():
            # OCR 수행
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_image(img)

        matched = extract_relevant_sentences(text)
        if matched:
            result.append(f"[페이지 {page_num + 1}]\n" + "\n".join(matched))

    return "\n\n".join(result)

# 전체 폴더 처리 함수 (CSV 저장)
def process_pdf_folder_to_csv(folder_path, output_csv):
    ensure_dir(os.path.dirname(output_csv))

    headers = [
        "pdf_name", "관련내용", "공사_대분류", "공사_중분류", "공사_소분류",
        "공종_대분류", "공종_세부", "사고객체_대분류", "사고객체_세부",
        "장소_대분류", "장소_세부", "부위_대분류", "부위_세부",
        "인적사고", "물적사고", "작업프로세스", "사고원인"
    ]

    rows = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"분석 중: {filename}")
            extracted = extract_text_from_pdf(pdf_path)

            if extracted:
                auto = auto_classify(extracted, CATEGORY_MAP)
                rows.append([
                    filename, extracted,
                    auto.get("공사_대분류", ""), auto.get("공사_중분류", ""), auto.get("공사_소분류", ""),
                    auto.get("공종_대분류", ""), auto.get("공종_세부", ""), auto.get("사고객체_대분류", ""), auto.get("사고객체_세부", ""),
                    auto.get("장소_대분류", ""), auto.get("장소_세부", ""), auto.get("부위_대분류", ""), auto.get("부위_세부", ""),
                    auto.get("인적사고", ""), auto.get("물적사고", ""), auto.get("작업프로세스", ""), auto.get("사고원인", "")
                ])

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\n완료! 결과가 다음 위치에 저장되었습니다: {output_csv}")

if __name__ == "__main__":
    input_folder = "data/pdf"
    output_file = "code/JangHeon/result/classification.csv"
    process_pdf_folder_to_csv(input_folder, output_file)
