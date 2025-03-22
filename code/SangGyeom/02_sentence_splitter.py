import os
import pandas as pd
import re
from typing import List, Dict, Any
import glob


def split_into_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분리하는 함수
    
    Args:
        text (str): 분리할 텍스트
        
    Returns:
        List[str]: 분리된 문장 리스트
    """
    # 특수한 약어 패턴(예: F.C.M)을 임시 마커로 교체
    text = re.sub(r'([A-Z])\.([ ]*[A-Z])', r'\1<<DOT>>\2', text)
    
    # 영어 단어 사이의 점도 보존 (예: 3m 5m 등의 숫자+단위)
    text = re.sub(r'(\d+[a-zA-Z]+)\.(\d+[a-zA-Z]+)', r'\1<<DOT>>\2', text)
    
    # 문장의 실제 끝을 찾기 위한 패턴
    # 한글 문자 다음에 마침표가 오고, 그 뒤에 공백이나 문장의 끝이 오는 경우
    pattern = re.compile(r'([가-힣]\.)(?=\s|$)')
    marked_text = pattern.sub(r'\1<<SPLIT>>', text)
    
    # 임시 마커를 원래 문자로 복원
    marked_text = marked_text.replace('<<DOT>>', '.')
    
    # 분리 마커로 문장 나누기
    sentences = marked_text.split('<<SPLIT>>')
    
    # 빈 문장 제거 및 각 문장 정리
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    # 빈 문장이 없는 경우 원본 텍스트 반환
    if not sentences and text.strip():
        return [text.strip()]
    
    return sentences


def merge_incomplete_sentences(doc_df: pd.DataFrame) -> pd.DataFrame:
    """마침표가 없는 본문 텍스트를 다음 본문과 합치는 함수
    
    Args:
        doc_df (pd.DataFrame): 문서 데이터프레임
        
    Returns:
        pd.DataFrame: 처리된 데이터프레임
    """
    # 결과를 저장할 새 데이터프레임
    result_rows = []
    
    # 데이터프레임을 리스트로 변환하여 처리
    rows = doc_df.to_dict('records')
    
    i = 0
    while i < len(rows):
        current_row = rows[i].copy()
        
        # 현재 행이 본문이고 한글 문자 뒤에 마침표가 없거나 마침표 뒤에 공백이 없는 경우
        if current_row['context_type'] == '본문' and not re.search(r'[가-힣]\.(?=\s|$)', current_row['text']):
            # 합칠 텍스트 초기화
            merged_text = current_row['text']
            
            # 다음 행들을 확인하며 본문 합치기
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                
                # 다음 행이 본문인 경우 합치기
                if next_row['context_type'] == '본문':
                    # 공백 추가하여 텍스트 합치기
                    if merged_text and not merged_text.endswith(' ') and not next_row['text'].startswith(' '):
                        merged_text += ' ' + next_row['text']
                    else:
                        merged_text += next_row['text']
                    j += 1
                    
                    # 한글 문자 뒤에 마침표가 있고 그 뒤에 공백이나 문장 끝이 오면 합치기 종료
                    if re.search(r'[가-힣]\.(?=\s|$)', next_row['text']):
                        break
                else:
                    # 본문이 아닌 경우(소제목 등) 합치기 중단
                    break
            
            # 합쳐진 텍스트로 현재 행 업데이트
            current_row['text'] = merged_text
            result_rows.append(current_row)
            
            # 다음 처리할 행 인덱스 업데이트
            i = j
        else:
            # 일반적인 경우 현재 행 추가
            result_rows.append(current_row)
            i += 1
    
    # 결과를 데이터프레임으로 변환
    return pd.DataFrame(result_rows)


def process_data_to_sentences(input_csv: str, output_csv: str) -> None:
    """PDF 처리 결과를 문장 단위 데이터로 변환하는 함수
    
    Args:
        input_csv (str): 입력 CSV 파일 경로
        output_csv (str): 출력 CSV 파일 경로
    """
    print(f"데이터 처리 시작: {input_csv}")
    
    # 파일 존재 여부 확인
    if not os.path.exists(input_csv):
        print(f"오류: 입력 파일이 존재하지 않습니다 - {input_csv}")
        
        # results 폴더에서 개별 파일 찾기 시도
        result_files = glob.glob("./code/SangGyeom/results/*_data.csv")
        if result_files:
            print(f"개별 결과 파일 {len(result_files)}개를 찾았습니다. 이를 합쳐서 처리합니다.")
            
            # 모든 파일의 데이터 합치기
            dfs = []
            for file in result_files:
                try:
                    df = pd.read_csv(file, encoding='utf-8-sig')
                    dfs.append(df)
                    print(f"파일 로드: {file}, {df.shape[0]} 행")
                except Exception as e:
                    print(f"파일 로드 오류 {file}: {str(e)}")
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                print(f"데이터 합치기 완료: {df.shape[0]} 행, {df.shape[1]} 열")
            else:
                print("처리할 데이터가 없습니다.")
                return
        else:
            print("데이터를 찾을 수 없습니다. 먼저 01_pdf_preprocessing.py를 실행하세요.")
            return
    else:
        # CSV 파일 로드
        try:
            df = pd.read_csv(input_csv, encoding='utf-8-sig')
            print(f"원본 데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            return
    
    # 1페이지와 2페이지 데이터 제외
    df = df[~df['page'].isin([1, 2])]
    print(f"1페이지와 2페이지 제외 후 데이터: {df.shape[0]} 행")
    
    # 결과를 저장할 리스트
    processed_rows = []
    
    # 문서별로 그룹화하여 처리
    grouped = df.groupby('source')
    
    for doc_name, doc_df in grouped:
        print(f"문서 처리 중: {doc_name}")
        
        # 문서 내 라인을 행 번호 순서대로 정렬
        doc_df = doc_df.sort_values(by=['page', 'line_num'])
        
        # 완벽한 문장 처리를 위해 텍스트를 소제목별로 합치기
        subtitle_groups = {}
        
        # 소제목 플래그 초기화
        current_subtitle = None
        current_title = None
        
        # 먼저 모든 텍스트를 소제목별로 그룹화
        for _, row in doc_df.iterrows():
            context_type = row['context_type']
            text = row['text']
            
            # 제목 정보 저장
            if 'title' in row:
                current_title = row['title']
            
            # 소제목 업데이트
            if context_type == "소제목":
                current_subtitle = text
                # 소제목 그룹 초기화
                if current_subtitle not in subtitle_groups:
                    subtitle_groups[current_subtitle] = {
                        'texts': [],
                        'title': current_title,
                        'first_page': row['page'],
                        'first_line': row['line_num'],
                        'source': row['source']
                    }
                continue
            
            # 소제목이 등장하기 전의 데이터는 저장하지 않음
            if current_subtitle is None:
                continue
            
            # 본문 텍스트 저장
            if context_type == "본문":
                # 짧은 텍스트가 다음 행과 합쳐질 가능성이 있지만, 모든 텍스트를 수집
                subtitle_groups[current_subtitle]['texts'].append(text)
        
        # 각 소제목 그룹에 대해 문장 처리
        for subtitle, group_data in subtitle_groups.items():
            # 모든 텍스트를 하나로 합침
            full_text = ' '.join(group_data['texts'])
            
            # 불필요한 공백 제거
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # 문장 단위로 분리
            sentences = split_into_sentences(full_text)
            
            # 각 문장별로 행 생성
            for sentence in sentences:
                if sentence:  # 빈 문장 제외
                    processed_rows.append({
                        'title': group_data['title'],
                        'subtitle': subtitle,
                        'sentence': sentence,
                        'page': group_data['first_page'],
                        'line_num': group_data['first_line'],
                        'source': group_data['source']
                    })
    
    # 결과를 데이터프레임으로 변환
    result_df = pd.DataFrame(processed_rows)
    
    # 결과가 비어있는지 확인
    if result_df.empty:
        print("경고: 처리된 데이터가 없습니다.")
        return
    
    print(f"처리 결과: {result_df.shape[0]} 행, {result_df.shape[1]} 열")
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # CSV로 저장 - quoting=1은 따옴표가 필요한 필드에만 따옴표 적용
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig', quoting=1)
    print(f"결과 저장 완료: {output_csv}")


if __name__ == "__main__":
    # 파일 경로 설정
    input_csv = "./code/SangGyeom/data/processed_pdf_data.csv"
    output_csv = "./code/SangGyeom/data/processed_pdf_data2.csv"
    
    # 데이터 처리 및 저장
    process_data_to_sentences(input_csv, output_csv)
