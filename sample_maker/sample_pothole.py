"""
CSV 결과에서 확실한 포트홀 30개만 랜덤 샘플링해서 복사

사용법:
    python sample_pothole.py
"""

import pandas as pd
import shutil
from pathlib import Path
import random

# ============================================================================
# 설정
# ============================================================================

# CSV 파일 경로
CSV_PATH = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\inference_results\test_images_results_20260429_105659.csv"

# 원본 이미지 폴더
SOURCE_DIR = r"C:\Users\Dell3571\Downloads\archive (1)\images"

# 출력 폴더
OUTPUT_DIR = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\data\images\pothole"

# 샘플 개수
SAMPLE_COUNT = 400

# ============================================================================
# 메인 함수
# ============================================================================

def sample_potholes():
    """확실한 포트홀 이미지 30개 랜덤 샘플링"""
    
    print(f"\n{'='*80}")
    print(f"확실한 포트홀 샘플링 ({SAMPLE_COUNT}개)")
    print(f"{'='*80}\n")
    
    # CSV 로드
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    
    # 확실한 포트홀만 필터링
    definite_pothole = df[df["CP클래스"] == "확실한 포트홀"]
    
    print(f"📊 확실한 포트홀: {len(definite_pothole)}장")
    
    if len(definite_pothole) < SAMPLE_COUNT:
        print(f"⚠️  요청한 {SAMPLE_COUNT}개보다 적습니다. 전체 {len(definite_pothole)}개만 복사합니다.")
        sample_count = len(definite_pothole)
    else:
        sample_count = SAMPLE_COUNT
    
    # 랜덤 샘플링
    sampled = definite_pothole.sample(n=sample_count, random_state=42)
    
    print(f"🎲 랜덤 샘플링: {sample_count}개 선택\n")
    
    # 출력 폴더 생성
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    source_base = Path(SOURCE_DIR)
    
    # 복사
    print("📁 복사 중...")
    copied = 0
    
    for idx, (_, row) in enumerate(sampled.iterrows(), 1):
        filename = row["파일명"]
        src = source_base / filename
        dst = output_path / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
            print(f"  [{idx}/{sample_count}] {filename}")
        else:
            print(f"  ⚠️  파일 없음: {filename}")
    
    print(f"\n{'='*80}")
    print(f"✅ 완료!")
    print(f"{'='*80}\n")
    print(f"복사 완료: {copied}개")
    print(f"출력 위치: {output_path}\n")
    
    # 파일 목록 저장
    list_path = output_path.parent / "pothole_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        f.write(f"# 확실한 포트홀 샘플 {copied}개\n\n")
        for filename in sampled["파일명"]:
            f.write(f"{filename}\n")
    
    print(f"파일 목록 저장: {list_path}")


if __name__ == "__main__":
    sample_potholes()
