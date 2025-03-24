import pandas as pd
import csv

# 파일 경로 설정
# modified_path = "code\JinGyu\submission.csv"
# modified_path = "data\submission\JS\Jacode_submission.csv"
modified_path = "submission/final_submission_filled.csv"


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


# 5. 빈 문자열이 있는 행 인덱스 찾기
empty_string_rows = modified_df.applymap(lambda x: isinstance(x, str) and x.strip() == "").any(axis=1)
empty_row_indices = modified_df[empty_string_rows].index.tolist()

# 6. 결과 출력
print("\n✅ 빈 문자열을 포함한 행 인덱스:")
print(empty_row_indices)

print("\n✅ 해당 행 내용:")
print(modified_df.loc[empty_row_indices])


# if text_col in modified_df.columns:
#     modified_df[text_col] = modified_df[text_col].fillna("")
#     modified_df[text_col] = modified_df[text_col].apply(
#         lambda x: default_text if str(x).strip() == "" else x
#     )

# # 5. 결측값 재확인
# missing_info = modified_df.isnull().sum()
# print("남은 결측값:\n", missing_info[missing_info > 0])

# # 6. 저장
# modified_df.to_csv(output_path, index=False, encoding="utf-8-sig")
# print(f"저장 완료: {output_path}")