"""
현실적인 블랙박스 영상 생성 (2분 데모)

주행 시나리오:
- 빠른 정상 구간 (40초)
- 느린 정상 구간 (30초)
- 느린 포트홀 구간 (40초)
- 빠른 복귀 구간 (10초)

사용법:
    python create_demo_video.py
"""

import cv2
import numpy as np
from pathlib import Path
import random

# ============================================================================
# 설정
# ============================================================================

# 이미지 폴더 경로
BASE_DIR = Path(r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\data\images")

# 출력 영상 설정
OUTPUT_PATH = r"C:\Users\Dell3571\Desktop\PROJECTS\pothole-pipeline\sample_maker\demo_video.mp4"
FPS = 30
WIDTH = 1280
HEIGHT = 720

# 클래스별 색상 (BGR)
CLASS_COLORS = {
    'normal': (0, 255, 0),              # 초록 - 확실한 정상
    'uncertain_normal': (0, 255, 255),  # 노랑 - 애매 정상
    'uncertain_pothole': (0, 165, 255), # 주황 - 애매 포트홀
    'pothole': (0, 0, 255)              # 빨강 - 확실한 포트홀
}

CLASS_LABELS = {
    'normal': 'Normal (Confident)',
    'uncertain_normal': 'Normal (Uncertain)',
    'uncertain_pothole': 'Pothole (Uncertain)',
    'pothole': 'POTHOLE (Confident)'
}

# ============================================================================
# 시나리오 정의
# ============================================================================

def create_scenario():
    """
    2분 영상 시나리오 생성
    
    각 항목: (클래스명, 사용_이미지_개수, 프레임_per_이미지, 설명)
    """
    
    scenario = [
        # === 1. 빠른 정상 구간 (40초 = 1200프레임) ===
        ('normal', 280, 2, '빠른 정상 도로 주행'),
        ('pothole', 1, 2, '포트홀 스쳐감 1'),
        ('normal', 100, 2, '정상 도로'),
        ('uncertain_normal', 15, 2, '애매 정상'),
        ('normal', 80, 2, '정상 도로'),
        ('pothole', 1, 2, '포트홀 스쳐감 2'),
        ('normal', 100, 2, '정상 도로'),
        
        # === 2. 느린 정상 구간 (30초 = 900프레임) ===
        ('normal', 40, 6, '느린 정상 구간 진입'),
        ('uncertain_normal', 20, 5, '애매한 노면'),
        ('normal', 30, 6, '정상 유지'),
        
        # === 3. 느린 포트홀 구간 (40초 = 1200프레임) ===
        ('normal', 15, 5, '포트홀 구간 진입'),
        ('pothole', 1, 8, '포트홀 그룹1-1'),
        ('pothole', 1, 7, '포트홀 그룹1-2'),
        ('pothole', 1, 7, '포트홀 그룹1-3'),
        ('normal', 8, 6, '정상'),
        ('uncertain_pothole', 10, 5, '애매 포트홀들'),
        ('normal', 10, 6, '정상'),
        ('pothole', 1, 8, '포트홀 그룹2-1'),
        ('pothole', 1, 7, '포트홀 그룹2-2'),
        ('pothole', 1, 7, '포트홀 그룹2-3'),
        ('pothole', 1, 6, '포트홀 그룹2-4'),
        ('normal', 10, 6, '정상'),
        ('uncertain_normal', 10, 5, '애매 정상'),
        ('normal', 8, 6, '정상'),
        ('pothole', 1, 7, '포트홀 그룹3-1'),
        ('pothole', 1, 7, '포트홀 그룹3-2'),
        ('pothole', 1, 6, '포트홀 그룹3-3'),
        ('normal', 15, 6, '구간 빠져나옴'),
        
        # === 4. 빠른 복귀 구간 (10초 = 300프레임) ===
        ('normal', 150, 2, '빠른 정상 복귀'),
    ]
    
    return scenario


# ============================================================================
# 유틸리티 함수
# ============================================================================

def load_images_from_folder(folder_path):
    """폴더에서 이미지 경로 로드"""
    extensions = ['*.png', '*.jpg', '*.jpeg']
    images = []
    for ext in extensions:
        images.extend(list(folder_path.glob(ext)))
    return images


def resize_and_pad(image, target_width, target_height):
    """이미지를 비율 유지하며 리사이즈 후 패딩"""
    h, w = image.shape[:2]
    
    # 비율 계산
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 리사이즈
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 검은색 배경 생성
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 중앙 배치
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result


def add_overlay(frame, class_name, frame_number, total_frames):
    """프레임에 오버레이 추가"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # 상태 표시줄 배경 (상단)
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    
    # 클래스 라벨
    label = CLASS_LABELS[class_name]
    color = CLASS_COLORS[class_name]
    
    cv2.putText(overlay, label, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # 포트홀일 경우 경고 추가
    if 'pothole' in class_name:
        cv2.putText(overlay, "CAUTION!", (w - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # 프레임 정보 (하단)
    time_str = f"Time: {frame_number / FPS:.1f}s / {total_frames / FPS:.0f}s"
    cv2.putText(overlay, time_str, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 블렌딩
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    return frame


# ============================================================================
# 메인 함수
# ============================================================================

def create_demo_video():
    """데모 영상 생성"""
    
    print(f"\n{'='*80}")
    print(f"블랙박스 데모 영상 생성")
    print(f"{'='*80}\n")
    
    # 1. 이미지 로드
    print("📁 이미지 로드 중...")
    image_pools = {}
    for class_name in ['normal', 'uncertain_normal', 'uncertain_pothole', 'pothole']:
        folder = BASE_DIR / class_name
        images = load_images_from_folder(folder)
        image_pools[class_name] = images
        print(f"  {class_name}: {len(images)}장")
    
    # 2. 시나리오 생성
    print("\n🎬 시나리오 생성 중...")
    scenario = create_scenario()
    
    # 3. 프레임 시퀀스 생성
    print("🎞️  프레임 시퀀스 생성 중...")
    frame_sequence = []
    
    for class_name, num_images, frames_per_image, description in scenario:
        available_images = image_pools[class_name]
        
        if len(available_images) == 0:
            print(f"  ⚠️  {class_name} 이미지 없음, 건너뜀")
            continue
        
        # 랜덤 샘플링 (중복 허용 안 함)
        if num_images > len(available_images):
            num_images = len(available_images)
        
        selected_images = random.sample(available_images, num_images)
        
        # 각 이미지를 지정된 프레임 수만큼 반복
        for img_path in selected_images:
            for _ in range(frames_per_image):
                frame_sequence.append((img_path, class_name))
        
        print(f"  {description}: {num_images}개 × {frames_per_image}프레임 = {num_images * frames_per_image}프레임")
    
    total_frames = len(frame_sequence)
    duration = total_frames / FPS
    
    print(f"\n총 프레임: {total_frames}")
    print(f"영상 길이: {duration:.1f}초 ({duration/60:.1f}분)")
    
    # 4. 영상 생성
    print(f"\n🎥 영상 생성 중: {OUTPUT_PATH}\n")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))
    
    for i, (img_path, class_name) in enumerate(frame_sequence):
        # 이미지 로드
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"  ⚠️  이미지 로드 실패: {img_path}")
            continue
        
        # 리사이즈 & 패딩
        frame = resize_and_pad(img, WIDTH, HEIGHT)
        
        # 오버레이 추가
        frame = add_overlay(frame, class_name, i, total_frames)
        
        # 프레임 쓰기
        out.write(frame)
        
        # 진행상황 표시
        if (i + 1) % 100 == 0:
            progress = (i + 1) / total_frames * 100
            print(f"  진행: {i+1}/{total_frames} ({progress:.1f}%)")
    
    out.release()
    
    print(f"\n{'='*80}")
    print(f"✅ 완료!")
    print(f"{'='*80}\n")
    print(f"출력 파일: {OUTPUT_PATH}")
    print(f"해상도: {WIDTH}x{HEIGHT}")
    print(f"FPS: {FPS}")
    print(f"길이: {duration:.1f}초 ({duration/60:.1f}분)\n")


if __name__ == "__main__":
    create_demo_video()
