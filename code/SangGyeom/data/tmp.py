import pandas as pd

# CSV 파일 읽기
file_path = r"code\SangGyeom\data\submission0322_1_raw.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 컬럼 이름 가져오기
columns = list(df.columns)

# 두번째 컬럼명 수정
columns[1] = "재발방지대책 및 향후조치계획"

# 3번째 컬럼부터 마지막 컬럼까지 vec_0, vec_1, ... 으로 수정
for i in range(2, len(columns)):
    columns[i] = f"vec_{i-2}"

# 컬럼명 적용
df.columns = columns

# 수정된 DataFrame을 CSV 파일로 저장
df.to_csv(file_path.replace('.csv', '_modified.csv'), index=False, encoding='utf-8-sig')

print("파일이 성공적으로 수정되었습니다.")


# import pandas as pd

# # 파일 경로
# file1 = r"code\SangGyeom\data\submission0322_1_raw_modified.csv"
# file2 = r"code\SangGyeom\fake_submission.csv"

# # 파일 로드
# df1 = pd.read_csv(file1, encoding='utf-8-sig')
# df2 = pd.read_csv(file2, encoding='utf-8-sig')

# # 기본 정보 비교
# print("=== 파일 1 정보 ===")
# print(f"행 수: {len(df1)}")
# print(f"열 수: {len(df1.columns)}")
# print(f"컬럼 이름: {list(df1.columns)[:10]}")
# print(f"데이터 타입:\n{df1.dtypes}")
# print("\n")

# print("=== 파일 2 정보 ===")
# print(f"행 수: {len(df2)}")
# print(f"열 수: {len(df2.columns)}")
# print(f"컬럼 이름: {list(df2.columns)[:10]}")
# print(f"데이터 타입:\n{df2.dtypes}")

# # 컬럼 이름 비교
# if list(df1.columns) != list(df2.columns):
#     print("\n컬럼 이름이 다릅니다!")
#     # 차이점 출력
#     in_df1_not_in_df2 = set(df1.columns) - set(df2.columns)
#     in_df2_not_in_df1 = set(df2.columns) - set(df1.columns)
    
#     if in_df1_not_in_df2:
#         print(f"파일 1에만 있는 컬럼: {in_df1_not_in_df2}")
#     if in_df2_not_in_df1:
#         print(f"파일 2에만 있는 컬럼: {in_df2_not_in_df1}")
# else:
#     print("\n컬럼 이름이 동일합니다")

# # 데이터 샘플 확인 (첫 5개 행)
# print("\n=== 파일 1 첫 5개 행 ===")
# print(df1.head())
# print("\n=== 파일 2 첫 5개 행 ===")
# print(df2.head())

# import pandas as pd
# import csv

# # 파일 경로 설정
# fake_path = r"code\SangGyeom\fake_submission.csv"
# modified_path = r"code\SangGyeom\data\submission0322_1_raw_modified.csv"
# output_path = r"code\SangGyeom\data\submission0322_1_raw_modified_2.csv"

# # 1. 파일 로드
# fake_df = pd.read_csv(fake_path, encoding="utf-8-sig")
# modified_df = pd.read_csv(modified_path, encoding="utf-8-sig")

# # 2. ID 기준 정렬
# fake_df = fake_df.sort_values("ID").reset_index(drop=True)
# modified_df = modified_df.sort_values("ID").reset_index(drop=True)

# # 3. 컬럼 순서 맞추기
# modified_df = modified_df[fake_df.columns]

# # 4. "재발방지대책 및 향후조치계획" 컬럼 따옴표 감싸기
# text_col = "재발방지대책 및 향후조치계획"

# if text_col in modified_df.columns:
#     def quote_wrap(x):
#         if pd.isna(x):
#             return x
#         x_str = str(x).strip().strip('"')  # 기존 따옴표 제거
#         return f'"{x_str}"'
    
#     modified_df[text_col] = modified_df[text_col].apply(quote_wrap)

# # 5. 데이터 타입 맞추기
# for col in fake_df.columns:
#     target_dtype = fake_df[col].dtype
#     try:
#         modified_df[col] = modified_df[col].astype(target_dtype)
#     except Exception as e:
#         print(f"⚠️ 컬럼 '{col}' 타입 변환 실패: {e}")

# # 6. 저장
# modified_df.to_csv(
#     output_path,
#     index=False,
#     encoding="utf-8-sig",
#     quoting=csv.QUOTE_MINIMAL  # 콤마 포함 시만 따옴표 자동 추가
# )

# print(f"✅ 제출용 파일 저장 완료: {output_path}")

import pandas as pd
import csv

# 파일 경로 설정
modified_path = r"code\SangGyeom\data\submission0322_1_raw_modified.csv"
output_path = r"code\SangGyeom\data\submission0322_1_raw_modified_2.csv"


# 1. 파일 로드
modified_df = pd.read_csv(modified_path, encoding="utf-8-sig")

# 2. 결측값 확인
missing_info = modified_df.isnull().sum()
print("결측값이 있는 컬럼:\n", missing_info[missing_info > 0])

# 3. 일반적인 결측치 처리
for col in modified_df.columns:
    if modified_df[col].isnull().any():
        modified_df[col] = modified_df[col].fillna("" if modified_df[col].dtype == "object" else 0)

# 4. 재발방지대책 및 향후조치계획 컬럼 특수 처리
default_text = "작업자 안전교육 실시를 통한 재발 방지 대책 및 향후 조치 계획."
text_col = "재발방지대책 및 향후조치계획"

empty_string_cols = (modified_df.applymap(lambda x: isinstance(x, str) and x.strip() == "").sum().loc[lambda s: s > 0])

# 결과 출력
print("✅ 빈 문자열이 포함된 컬럼:")
print(empty_string_cols)

if text_col in modified_df.columns:
    modified_df[text_col] = modified_df[text_col].fillna("")
    modified_df[text_col] = modified_df[text_col].apply(
        lambda x: default_text if str(x).strip() == "" else x
    )

# 5. 결측값 재확인
missing_info = modified_df.isnull().sum()
print("남은 결측값:\n", missing_info[missing_info > 0])

# 6. 저장
modified_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"저장 완료: {output_path}")









