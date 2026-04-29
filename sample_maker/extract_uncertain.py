"""
CSV 결과를 기반으로 애매한 이미지들만 따로 복사하는 스크립트

사용법:
    python extract_uncertain.py
"""

import pandas as pd
import shutil
from pathlib import Path

# ============================================================================
# 설정
# ============================================================================

# CSV 파일 경로 (inference 결과)
CSV_PATH = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\inference_results\test_images_results_20260429_105659.csv"

# 원본 이미지 폴더
SOURCE_DIR = r"C:\Users\Dell3571\Downloads\archive (1)\images"

# 출력 폴더
OUTPUT_DIR = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\uncertain_images"

# ============================================================================
# 메인 함수
# ============================================================================

def extract_uncertain_images():
    """애매한 이미지들만 복사"""
    
    print(f"\n{'='*80}")
    print(f"애매한 이미지 추출")
    print(f"{'='*80}\n")
    
    # CSV 로드
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    
    print(f"총 {len(df)}개 이미지 분석 완료\n")
    
    # 애매한 이미지 필터링
    uncertain_pothole = df[df["CP클래스"] == "애매 (포트홀 쪽)"]
    uncertain_normal = df[df["CP클래스"] == "애매 (정상 쪽)"]
    
    print(f"📊 분류 결과:")
    print(f"  애매 (포트홀 쪽): {len(uncertain_pothole)}장")
    print(f"  애매 (정상 쪽):   {len(uncertain_normal)}장")
    print(f"  총 애매한 이미지: {len(uncertain_pothole) + len(uncertain_normal)}장\n")
    
    # 출력 폴더 생성
    output_base = Path(OUTPUT_DIR)
    pothole_dir = output_base / "uncertain_pothole"
    normal_dir = output_base / "uncertain_normal"
    
    pothole_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    source_base = Path(SOURCE_DIR)
    
    # 애매 포트홀 복사
    print("🔍 애매 (포트홀 쪽) 복사 중...")
    copied_pothole = 0
    for _, row in uncertain_pothole.iterrows():
        filename = row["파일명"]
        src = source_base / filename
        dst = pothole_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            copied_pothole += 1
        else:
            print(f"  ⚠️  파일 없음: {filename}")
    
    print(f"  ✅ {copied_pothole}장 복사 완료\n")
    
    # 애매 정상 복사
    print("🔍 애매 (정상 쪽) 복사 중...")
    copied_normal = 0
    for _, row in uncertain_normal.iterrows():
        filename = row["파일명"]
        src = source_base / filename
        dst = normal_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            copied_normal += 1
        else:
            print(f"  ⚠️  파일 없음: {filename}")
    
    print(f"  ✅ {copied_normal}장 복사 완료\n")
    
    # 파일 목록 저장
    print("📝 파일 목록 저장 중...")
    
    with open(output_base / "uncertain_pothole_list.txt", "w", encoding="utf-8") as f:
        f.write("# 애매 (포트홀 쪽)\n\n")
        for filename in uncertain_pothole["파일명"]:
            f.write(f"{filename}\n")
    
    with open(output_base / "uncertain_normal_list.txt", "w", encoding="utf-8") as f:
        f.write("# 애매 (정상 쪽)\n\n")
        for filename in uncertain_normal["파일명"]:
            f.write(f"{filename}\n")
    
    print(f"{'='*80}")
    print(f"✅ 완료!")
    print(f"{'='*80}\n")
    print(f"출력 위치:")
    print(f"  📁 {pothole_dir}")
    print(f"  📁 {normal_dir}\n")


if __name__ == "__main__":
    extract_uncertain_images()
