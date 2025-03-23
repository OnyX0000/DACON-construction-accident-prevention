import pandas as pd
import numpy as np
import json

# CSV 불러오기
csv_path = "data/test.csv"
df = pd.read_csv(csv_path)

# --------- CSV 전처리 코드 ---------
def preprocess_construction_data(df):
    # 날짜 및 시간 분리
    df[['날짜', '오전오후', '시간']] = df['발생일시'].str.extract(r'(\d{4}-\d{2}-\d{2}) (오전|오후) (\d{2}:\d{2})')

    # 기온 및 습도 숫자 추출
    df['기온(℃)'] = df['기온'].str.extract(r'(-?\d+\.?\d*)').astype(float)
    df['습도(%)'] = df['습도'].str.extract(r'(\d+\.?\d*)').astype(float)

    # 공사 종류 분할
    df[['공사_대분류', '공사_중분류', '공사_소분류']] = df['공사종류'].str.split(' / ', expand=True)

    # 층 정보 분할
    df['지상층'] = df['층 정보'].str.extract(r'지상\s*(\d+)층')[0].astype(float)
    df['지하층'] = df['층 정보'].str.extract(r'지하\s*(\d+)층')[0].astype(float)

    # 공종 분리
    df[['공종_대분류', '공종_세부']] = df['공종'].str.split(' > ', expand=True)

    # 사고객체, 장소, 부위 분리
    df[['사고객체_대분류', '사고객체_세부']] = df['사고객체'].str.split(' > ', n=1, expand=True)
    df[['장소_대분류', '장소_세부']] = df['장소'].str.split(' / ', n=1, expand=True)
    df[['부위_대분류', '부위_세부']] = df['부위'].str.split(' / ', n=1, expand=True)

    # 연면적 숫자화
    df['연면적(m2)'] = (
        df['연면적']
        .str.replace(',', '', regex=False)
        .str.replace('㎡', '', regex=False)
        .replace('-', np.nan)
        .astype(float)
    )

    return df

# 전처리 수행
df_clean = preprocess_construction_data(df)

# --------- metadata_categories.json 생성 코드 ---------
metadata_columns = [
    '공사_대분류', '공사_중분류', '공사_소분류',
    '지상층', '지하층',
    '공종_대분류', '공종_세부',
    '사고객체_대분류', '사고객체_세부',
    '장소_대분류', '장소_세부',
    '부위_대분류', '부위_세부',
    '인적사고', '물적사고', '작업프로세스'
]

unique_metadata_values = {}
for col in metadata_columns:
    if col in df_clean.columns:
        values = df_clean[col].dropna().unique().tolist()
        if df_clean[col].dtype == 'float':
            values = [str(int(v)) for v in values if not pd.isna(v)]
        unique_metadata_values[col] = list(set(values))

# 저장
metadata_path = "data/metadata_categories_test.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(unique_metadata_values, f, ensure_ascii=False, indent=2)

metadata_path

# 저장할 파일 경로
output_path = "data/test_with_metadata.csv"

# 필요한 컬럼만 포함해서 저장 (기존 컬럼 + 메타데이터 컬럼들)
save_columns = list(df.columns) + [col for col in metadata_columns if col in df_clean.columns]
df_clean.to_csv(output_path, columns=save_columns, index=False, encoding="utf-8-sig")

print(f"✅ 저장 완료: {output_path}")
