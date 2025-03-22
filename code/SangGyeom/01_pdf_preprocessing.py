import os
import re
import pdfplumber
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple, Dict, Any


def clean_text(text: str) -> str:
    """텍스트 정제 함수

    Args:
        text (str): 정제할 텍스트

    Returns:
        str: 정제된 텍스트
    """
    # 괄호 안의 내용 제거
    text = re.sub(r'\([^)]*\)', '', text)
    # 특수문자 제거
    text = re.sub(r'[^\w\s\-.,?!]', '', text)
    # 여러 공백을 하나로 변환
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


def extract_lines_with_font_sizes(pdf_path: str) -> List[Tuple[str, float, int, int]]:
    """PDF에서 텍스트와 폰트 크기, 페이지 번호, 라인 번호를 추출하는 함수

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        List[Tuple[str, float, int, int]]: (텍스트, 폰트 크기, 페이지 번호, 라인 번호) 리스트
    """
    result = []
    total_line_num = 0  # 전체 라인 번호 카운터
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 페이지에서 텍스트 추출
                try:
                    # 라인별로 단어 그룹화를 위한 딕셔너리
                    lines_dict = {}
                    
                    # extra_attrs 매개변수를 추가하여 정확한 폰트 정보 추출
                    words = page.extract_words(x_tolerance=1, y_tolerance=1, extra_attrs=["fontname", "size", "top"])
                    
                    for word in words:
                        if word['text'].strip():
                            # 'size' 키가 없는 경우 기본값 12.0 사용
                            try:
                                font_size = word.get('size', 12.0)
                                # size가 None이면 기본값 적용
                                if font_size is None:
                                    font_size = 12.0
                            except:
                                font_size = 12.0
                                
                            top = word.get('top', 0)  # y-좌표
                            
                            # top 위치를 기준으로 같은 라인끼리 그룹화
                            line_key = round(top, 0)  # 반올림하여 유사한 y 위치를 동일 라인으로 처리
                            
                            if line_key not in lines_dict:
                                lines_dict[line_key] = {"text": word['text'], "size": font_size}
                            else:
                                lines_dict[line_key]["text"] += " " + word['text']
                                # 라인의 폰트 크기는 첫 번째 단어의 폰트 크기를 기준으로 유지
                    
                    # 라인을 y-좌표 기준으로 정렬
                    sorted_lines = sorted(lines_dict.items())
                    
                    # 정렬된 라인을 결과에 추가
                    for _, line_data in sorted_lines:
                        total_line_num += 1  # 라인 번호 증가
                        result.append((line_data["text"], line_data["size"], page_num, total_line_num))
                
                except Exception as e:
                    print(f"페이지 {page_num} 처리 오류: {str(e)}")
                    # 오류 발생 시 대안 방법으로 텍스트 추출 시도
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            for line in page_text.split('\n'):
                                if line.strip():
                                    total_line_num += 1
                                    # 텍스트가 추출되면 기본 폰트 크기 12.0으로 저장
                                    result.append((line, 12.0, page_num, total_line_num))
                    except:
                        pass
        
        print(f"PDF 파일에서 {len(result)}개의 텍스트 라인 추출: {pdf_path}")
        
        # 폰트 크기가 없거나 0인 경우 기본값으로 설정
        for i in range(len(result)):
            text, font_size, page_num, line_num = result[i]
            if font_size is None or font_size <= 0:
                result[i] = (text, 12.0, page_num, line_num)
    
    except Exception as e:
        print(f"PDF 처리 오류: {pdf_path} - {str(e)}")
    
    return result


def estimate_font_sizes(lines_with_fonts: List[Tuple[str, float, int, int]]) -> Dict[str, float]:
    """문서의 폰트 크기 통계를 계산하는 함수

    Args:
        lines_with_fonts (List[Tuple[str, float, int, int]]): (텍스트, 폰트 크기, 페이지 번호, 라인 번호) 리스트

    Returns:
        Dict[str, float]: 폰트 크기 통계 (최대, 최소, 평균, 중간값)
    """
    # 유효한 폰트 크기만 필터링
    font_sizes = [size for _, size, _, _ in lines_with_fonts if size is not None and size > 0]
    
    # 폰트 크기 개수 출력 (디버깅용)
    print(f"유효한 폰트 크기 개수: {len(font_sizes)}")
    if font_sizes:
        print(f"폰트 크기 범위: {min(font_sizes)} ~ {max(font_sizes)}")
    
    if not font_sizes:
        print("경고: 폰트 크기 정보가 없습니다. 기본값을 사용합니다.")
        return {
            'max': 12.0,  # 기본값 설정
            'min': 10.0,
            'avg': 12.0,
            'median': 12.0
        }
    
    return {
        'max': max(font_sizes),
        'min': min(font_sizes),
        'avg': sum(font_sizes) / len(font_sizes),
        'median': sorted(font_sizes)[len(font_sizes) // 2]
    }


def determine_font_contexts(lines_with_fonts: List[Tuple[str, float, int, int]]) -> Dict[str, float]:
    """폰트 크기 빈도 분석을 통해 본문과 소제목 폰트 크기를 결정

    Args:
        lines_with_fonts (List[Tuple[str, float, int, int]]): (텍스트, 폰트 크기, 페이지 번호, 라인 번호) 리스트

    Returns:
        Dict[str, float]: 본문과 소제목 폰트 크기
    """
    # 본문 후보 (글자 수 20자 이상) 라인의 폰트 크기 수집
    body_candidate_sizes = []
    
    # 모든 라인의 폰트 크기를 소수점 첫째자리로 반올림
    rounded_lines_with_fonts = []
    for text, font_size, page_num, line_num in lines_with_fonts:
        rounded_font_size = round(font_size, 1) if font_size is not None and font_size > 0 else font_size
        rounded_lines_with_fonts.append((text, rounded_font_size, page_num, line_num))
    
    for text, font_size, _, _ in rounded_lines_with_fonts:
        if len(text) >= 20 and font_size is not None and font_size > 0:
            body_candidate_sizes.append(font_size)
    
    if not body_candidate_sizes:
        print("경고: 본문 후보가 없습니다. 기본값을 사용합니다.")
        return {
            'body_font': 12.0,
            'title_font': 14.0,
        }
    
    # 폰트 크기 빈도 계산
    size_frequency = {}
    for size in body_candidate_sizes:
        size_frequency[size] = size_frequency.get(size, 0) + 1
    
    # 가장 빈도가 높은 폰트 크기를 본문 폰트로 결정
    body_font_size = max(size_frequency.items(), key=lambda x: x[1])[0]
    
    # 본문보다 큰 폰트 크기들 중 최소값을 소제목 폰트로 결정
    larger_sizes = [size for size in set(s[1] for s in rounded_lines_with_fonts) if size > body_font_size]
    
    if larger_sizes:
        title_font_size = min(larger_sizes)
    else:
        title_font_size = body_font_size * 1.2  # 소제목 폰트가 없으면 본문의 1.2배로 설정
    
    print(f"분석 결과: 본문 폰트 크기 = {body_font_size}, 소제목 폰트 크기 = {title_font_size}")
    
    return {
        'body_font': body_font_size,
        'title_font': title_font_size
    }


def classify_line_context(text: str, font_size: float, font_contexts: Dict[str, float]) -> str:
    """라인의 폰트 크기를 기준으로 컨텍스트 유형 분류

    Args:
        text (str): 라인 텍스트
        font_size (float): 폰트 크기
        font_contexts (Dict[str, float]): 폰트 컨텍스트 정보 (본문/소제목 폰트 크기)

    Returns:
        str: 컨텍스트 유형 (소제목, 본문, 기타)
    """
    # 반올림된 폰트 크기
    rounded_size = round(font_size, 1)
    
    # 한글이 포함되어 있는지 확인
    has_korean = re.search(r'[가-힣]', text)
    
    if not has_korean:
        return "기타"  # 한글이 없는 경우 (그림, 표, 영문 등)
    
    body_font = font_contexts['body_font']
    title_font = font_contexts['title_font']
    
    # 분류 규칙 적용
    if rounded_size >= title_font:
        return "소제목"  # 소제목 폰트 크기 이상
    elif rounded_size == body_font:
        return "본문"  # 본문 폰트 크기와 일치
    else:
        return "기타"  # 나머지는 기타로 분류


def process_extracted_lines(lines_with_fonts: List[Tuple[str, float, int, int]], 
                            font_stats: Dict[str, float], 
                            filename: str) -> List[Dict[str, Any]]:
    """추출된 텍스트 라인별로 정보를 처리하여 DataFrame 형태로 반환

    Args:
        lines_with_fonts (List[Tuple[str, float, int, int]]): (텍스트, 폰트 크기, 페이지 번호, 라인 번호) 리스트
        font_stats (Dict[str, float]): 폰트 크기 통계
        filename (str): 원본 파일명

    Returns:
        List[Dict[str, Any]]: 처리된, 분류된 라인 정보
    """
    processed_lines = []
    
    # 폰트 크기가 없는 경우에도 처리할 수 있도록 함
    if not lines_with_fonts:
        processed_lines.append({
            'title': os.path.basename(filename),
            'text': "",
            'page': 1,
            'line_num': 1,
            'font_size': 12.0,
            'context_type': "기타",
            'source': filename
        })
        return processed_lines
    
    # 모든 라인의 폰트 크기를 소수점 첫째자리로 반올림
    rounded_lines_with_fonts = []
    for text, font_size, page_num, line_num in lines_with_fonts:
        rounded_font_size = round(font_size, 1) if font_size is not None and font_size > 0 else font_size
        rounded_lines_with_fonts.append((text, rounded_font_size, page_num, line_num))
    
    # 폰트 컨텍스트 결정 (본문과 소제목 폰트 크기)
    font_contexts = determine_font_contexts(rounded_lines_with_fonts)
    
    # 각 라인별로 처리
    for text, font_size, page_num, line_num in rounded_lines_with_fonts:
        # 텍스트 정제
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue
        
        # 컨텍스트 유형 분류
        context_type = classify_line_context(cleaned_text, font_size, font_contexts)
        
        # 결과 저장
        processed_lines.append({
            'title': os.path.basename(filename),
            'text': cleaned_text,
            'page': page_num,
            'line_num': line_num,
            'font_size': font_size,  # 이미 반올림된 값 저장
            'context_type': context_type,
            'source': filename
        })
    
    print(f"처리 결과: {len(processed_lines)}개의 라인 처리 완료")
    
    # 결과 통계 출력
    context_counts = {}
    for line in processed_lines:
        context_type = line['context_type']
        context_counts[context_type] = context_counts.get(context_type, 0) + 1
    
    print(f"컨텍스트 유형별 라인 수: {context_counts}")
    
    return processed_lines


def process_pdf_directory(pdf_dir: str) -> List[Dict[str, Any]]:
    """PDF 디렉토리 내의 모든 PDF 파일을 처리

    Args:
        pdf_dir (str): PDF 파일이 있는 디렉토리 경로

    Returns:
        List[Dict[str, Any]]: 모든 PDF에서 추출한 라인 정보 리스트
    """
    all_lines = []
    
    # 디렉토리 확인
    if not os.path.exists(pdf_dir):
        print(f"오류: 디렉토리가 존재하지 않습니다 - {pdf_dir}")
        return all_lines
    
    # 디렉토리 내 PDF 파일 목록 확인
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"오류: PDF 파일을 찾을 수 없습니다 - {pdf_dir}")
        return all_lines
    
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
    
    # 디렉토리 내 모든 PDF 파일 처리
    for filename in tqdm(pdf_files):
        file_path = os.path.join(pdf_dir, filename)
        file_lines = []  # 개별 파일의 라인들 저장
        
        try:
            # PDF에서 텍스트와 폰트 크기, 라인 번호 추출
            lines_with_fonts = extract_lines_with_font_sizes(file_path)
            
            # 폰트 크기 통계 계산
            font_stats = estimate_font_sizes(lines_with_fonts)
            print(f"폰트 통계: {font_stats}")
            
            # 텍스트 라인 처리 및 컨텍스트 유형 분류
            processed_lines = process_extracted_lines(lines_with_fonts, font_stats, filename)
            
            # 결과 리스트에 추가
            file_lines.extend(processed_lines)
            all_lines.extend(processed_lines)
            
            print(f"처리 완료: {filename}, 추출된 라인 수: {len(processed_lines)}")
            
            # 개별 파일별로 CSV 저장
            file_basename = os.path.splitext(os.path.basename(filename))[0]
            output_file_csv = f"./code/SangGyeom/data/{file_basename}_data.csv"
            save_to_csv(file_lines, output_file_csv)
            
        except Exception as e:
            print(f"오류 발생: {filename} - {str(e)}")
            # 오류 발생 시에도 파일명으로 빈 라인 생성하여 추가
            error_line = {
                'title': filename,
                'text': f"파일 처리 오류: {str(e)}",
                'page': 1,
                'line_num': 1,
                'font_size': 12.0,
                'context_type': "기타",
                'source': filename
            }
            file_lines.append(error_line)
            all_lines.append(error_line)
            
            # 오류가 발생해도 파일별 CSV 저장
            file_basename = os.path.splitext(os.path.basename(filename))[0]
            output_file_csv = f"./code/SangGyeom/results/{file_basename}_data.csv"
            save_to_csv(file_lines, output_file_csv)
    
    print(f"총 {len(all_lines)}개의 라인 정보 생성 완료")
    return all_lines


def save_to_csv(lines: List[Dict[str, Any]], output_path: str) -> None:
    """처리된 라인 정보를 CSV 파일로 저장

    Args:
        lines (List[Dict[str, Any]]): 라인 정보 리스트
        output_path (str): 저장할 CSV 파일 경로
    """
    if not lines:
        print("경고: 저장할 데이터가 없습니다.")
        return
    
    # 데이터프레임 변환 전 데이터 확인
    print(f"CSV 저장 전 데이터 확인: {len(lines)}개 항목")
    
    df = pd.DataFrame(lines)
    print(f"데이터프레임 변환 완료: {df.shape}")
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSV 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"CSV 파일 저장 완료: {output_path}, 총 {len(lines)}개 라인")
    
    # 저장된 파일 확인
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"저장된 파일 크기: {file_size} 바이트")
    else:
        print(f"경고: 파일이 생성되지 않았습니다 - {output_path}")


if __name__ == "__main__":
    # 디렉토리 경로 설정
    pdf_dir = "./data/pdf/"
    output_csv = "./code/SangGyeom/data/processed_pdf_data.csv"
    
    # 절대 경로 출력
    print(f"PDF 디렉토리 절대 경로: {os.path.abspath(pdf_dir)}")
    print(f"출력 CSV 절대 경로: {os.path.abspath(output_csv)}")
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # PDF 처리
    lines = process_pdf_directory(pdf_dir)
    
    # 통합 CSV로 저장
    save_to_csv(lines, output_csv)
    