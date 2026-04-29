"""
CSV 결과를 읽고 CP 클래스별로 이미지를 각 폴더로 분배

사용법:
    python distribute_images.py
"""

import pandas as pd
import shutil
from pathlib import Path

# ============================================================================
# 설정
# ============================================================================

# CSV 파일 경로
CSV_PATH = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\inference_results\test_images_results_20260429_110733.csv"

# 원본 이미지 폴더
SOURCE_DIR = r"C:\Users\Dell3571\Downloads\New_sample\원천데이터\2.CRACK\2.Highway\C_Highway_G01"

# 출력 폴더 (sample_maker 기준)
BASE_DIR = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\data\images"

# CP 클래스별 폴더 매핑
CLASS_FOLDERS = {
    0: "확실한_정상",           # 확실한 정상
    1: "애매_정상_쪽",          # 애매 (정상 쪽)
    2: "애매_포트홀_쪽",        # 애매 (포트홀 쪽)
    3: "pothole"              # 확실한 포트홀 (기존 폴더명 유지)
}

# ============================================================================
# 메인 함수
# ============================================================================

def distribute_images():
    """CSV 결과를 읽고 이미지를 CP 클래스별로 분배"""
    
    print(f"\n{'='*80}")
    print(f"이미지 분배 시작")
    print(f"{'='*80}\n")
    
    # CSV 로드
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    
    print(f"총 {len(df)}개 이미지 분석 완료\n")
    
    # CP 클래스별 통계
    class_counts = df["CP클래스번호"].value_counts().sort_index()
    print("📊 CP 클래스별 분포:")
    for class_num, count in class_counts.items():
        class_name = CLASS_FOLDERS.get(class_num, "알 수 없음")
        print(f"  Class {class_num} ({class_name}): {count}장")
    print()
    
    # 출력 폴더 생성
    base_path = Path(BASE_DIR)
    for class_num, folder_name in CLASS_FOLDERS.items():
        folder_path = base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
    
    source_base = Path(SOURCE_DIR)
    
    # CP 클래스별로 이미지 복사
    copied_counts = {class_num: 0 for class_num in CLASS_FOLDERS.keys()}
    
    print("📁 이미지 복사 중...\n")
    
    for _, row in df.iterrows():
        filename = row["파일명"]
        cp_class = row["CP클래스번호"]
        
        if cp_class not in CLASS_FOLDERS:
            print(f"  ⚠️  알 수 없는 클래스: {cp_class} ({filename})")
            continue
        
        src = source_base / filename
        dst_folder = base_path / CLASS_FOLDERS[cp_class]
        dst = dst_folder / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            copied_counts[cp_class] += 1
        else:
            print(f"  ⚠️  파일 없음: {filename}")
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"✅ 완료!")
    print(f"{'='*80}\n")
    
    print("복사 결과:")
    for class_num, folder_name in CLASS_FOLDERS.items():
        count = copied_counts[class_num]
        print(f"  {folder_name}: {count}장")
        print(f"    → {base_path / folder_name}")
    
    total_copied = sum(copied_counts.values())
    print(f"\n총 {total_copied}장 복사 완료\n")


if __name__ == "__main__":
    distribute_images()
