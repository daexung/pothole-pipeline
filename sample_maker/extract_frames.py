"""
영상의 모든 프레임을 이미지로 추출

30fps 영상 → 모든 프레임 추출
추출된 이미지는 inference.py로 테스트 가능

사용법:
    python extract_frames.py
"""

import cv2
from pathlib import Path

# ============================================================================
# 설정
# ============================================================================

# 입력 영상 경로
VIDEO_PATH = r"C:\Users\Dell3571\Downloads\13588981_3840_2160_30fps.mp4"

# 출력 폴더 경로
OUTPUT_DIR = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\extracted_frames"

# ============================================================================
# 메인 함수
# ============================================================================

def extract_frames():
    """영상에서 모든 프레임 추출"""
    
    print(f"\n{'='*80}")
    print(f"영상 프레임 추출 (모든 프레임)")
    print(f"{'='*80}\n")
    
    # 출력 폴더 생성
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 영상 열기
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ 영상 파일을 열 수 없습니다: {VIDEO_PATH}")
        return
    
    # 영상 정보
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 원본 영상 정보:")
    print(f"  경로: {VIDEO_PATH}")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  해상도: {width}x{height}")
    print(f"  총 프레임: {total_frames}")
    print(f"  길이: {duration:.2f}초 ({duration/60:.2f}분)\n")
    
    print(f"🎯 추출 설정:")
    print(f"  모든 프레임 추출")
    print(f"  예상 추출 이미지: {total_frames}장\n")
    
    # 프레임 추출
    print(f"📸 프레임 추출 중...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 모든 프레임 저장
        filename = f"frame_{frame_count:04d}.png"
        filepath = output_path / filename
        
        cv2.imwrite(str(filepath), frame)
        frame_count += 1
        
        # 진행상황 표시
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  진행: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"✅ 완료!")
    print(f"{'='*80}\n")
    print(f"총 추출 이미지: {frame_count}장")
    print(f"출력 폴더: {output_path}\n")
    print(f"📌 다음 단계:")
    print(f"1. inference_simple.py의 TEST_DIR 경로를 다음으로 변경:")
    print(f"   TEST_DIR = r\"{OUTPUT_DIR}\"")
    print(f"2. python inference_simple.py 실행\n")


if __name__ == "__main__":
    extract_frames()