"""
비디오 → 프레임 추출 → Inference 통합 파이프라인

사용법:
    python inference_advance.py \
        --video ./sample_video.mp4 \
        --model ./models/best.pt \
        --cpjson ./models/cp_results_90.json \
        --out ./inference_results
        
25fps 비디오라면:
    1초 = 25개 프레임 (모두 추출됨)
    결과를 CSV로 저장하여 fake log generator 설계에 활용 가능
"""

import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
from torchvision import transforms
import pandas as pd
import argparse

from config import create_model


# ============================================================
# 비디오에서 프레임 추출
# ============================================================
def extract_frames_from_video(video_path, output_folder=None):
    """
    비디오 파일에서 모든 프레임을 추출
    
    Args:
        video_path: 비디오 파일 경로
        output_folder: 프레임 저장 폴더 (None이면 메모리에만 유지)
    
    Returns:
        frames_info: [{'frame_num': int, 'timestamp': float, 'path': Path}, ...]
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*80}")
    print(f"🎬 비디오 정보")
    print(f"{'='*80}")
    print(f"파일: {Path(video_path).name}")
    print(f"해상도: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"총 프레임 수: {total_frames}")
    print(f"재생 시간: {total_frames/fps:.2f}초")
    print(f"{'='*80}\n")
    
    # 출력 폴더 생성
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    frames_info = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_num / fps  # 비디오 내 시간 (초)
        
        frame_data = {
            'frame_num': frame_num,
            'timestamp': timestamp,
            'frame': frame,
            'path': None
        }
        
        # 프레임을 파일로 저장
        if output_folder:
            frame_path = output_folder / f"frame_{frame_num:06d}_{timestamp:.3f}s.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_data['path'] = frame_path
        
        frames_info.append(frame_data)
        
        if (frame_num + 1) % 100 == 0 or (frame_num + 1) == total_frames:
            elapsed_sec = (frame_num + 1) / fps
            print(f"   {frame_num + 1}/{total_frames} 프레임 추출 완료 ({elapsed_sec:.2f}초)...")
        
        frame_num += 1
    
    cap.release()
    
    print(f"\n✅ 총 {len(frames_info)}개 프레임 추출 완료\n")
    
    return frames_info


# ============================================================
# 전처리 함수 (본네트 crop 포함)
# ============================================================
def preprocess(img_array_or_path, imgsz=224, crop_bonnet_ratio=0.2):
    """
    이미지 전처리 (본네트 영역 제거)
    
    Args:
        img_array_or_path: numpy array (BGR) 또는 이미지 파일 경로
        imgsz: 입력 크기
        crop_bonnet_ratio: 하단에서 제거할 비율 (0.2 = 20%)
    
    Returns:
        torch tensor (1, 3, 224, 224)
    """
    # numpy array (BGR) → PIL Image (RGB)
    if isinstance(img_array_or_path, np.ndarray):
        img_array = img_array_or_path.copy()
        
        # 본네트 영역 제거 (하단 crop_bonnet_ratio%)
        height = img_array.shape[0]
        crop_px = int(height * crop_bonnet_ratio)
        img_array = img_array[:height - crop_px, :]
        
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    else:
        img = Image.open(img_array_or_path).convert("RGB")
        # 파일에서 로드한 경우도 crop
        img_np = np.array(img)
        height = img_np.shape[0]
        crop_px = int(height * crop_bonnet_ratio)
        img_np = img_np[:height - crop_px, :]
        img = Image.fromarray(img_np)
    
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0)


# ============================================================
# CP Inference
# ============================================================
def run_inference_on_frames(frames_info, model_path, cp_json_path, 
                           output_csv, device="cuda:0", crop_bonnet_ratio=0.2):
    """
    모든 프레임에 대해 inference 실행
    
    Args:
        frames_info: extract_frames_from_video의 반환값
        model_path: 모델 파일 경로
        cp_json_path: CP results JSON 파일 경로
        output_csv: 결과 CSV 저장 경로
        device: "cuda:0" 또는 "cpu"
        crop_bonnet_ratio: 본네트 crop 비율
    
    Returns:
        pandas DataFrame
    """
    
    print(f"{'='*80}")
    print(f"🔍 Inference 시작")
    print(f"{'='*80}")
    print(f"📊 본네트 crop 비율: {crop_bonnet_ratio*100:.0f}%\n")
    
    # CP JSON 로드
    with open(cp_json_path, "r") as f:
        cp_data = json.load(f)
    
    quantiles = cp_data["cp_results"]["quantiles"]
    q0 = float(quantiles["0"])
    q1 = float(quantiles["1"])
    
    print(f"✅ CP Quantiles")
    print(f"   q0 (Normal):  {q0:.4f}")
    print(f"   q1 (Pothole): {q1:.4f}\n")
    
    # 모델 로드
    if not torch.cuda.is_available() and device == "cuda:0":
        device = "cpu"
        print(f"⚠️  GPU 사용 불가, CPU로 전환\n")
    
    model, _ = create_model("last_4_blocks", device)
    
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    
    model.eval()
    print(f"✅ 모델 로드 완료 (장치: {device})\n")
    
    records = []
    total = len(frames_info)
    
    print(f"🔍 {total}개 프레임 분석 중...\n")
    
    for idx, frame_data in enumerate(frames_info, 1):
        frame_num = frame_data['frame_num']
        timestamp = frame_data['timestamp']
        
        # 프레임 전처리 (본네트 crop 포함)
        if frame_data['path']:
            img_tensor = preprocess(frame_data['path'], crop_bonnet_ratio=crop_bonnet_ratio).to(device)
        else:
            img_tensor = preprocess(frame_data['frame'], crop_bonnet_ratio=crop_bonnet_ratio).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model.model(img_tensor)
            logits = logits[0] if isinstance(logits, tuple) else logits
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        p0, p1 = float(prob[0]), float(prob[1])
        s0, s1 = 1 - p0, 1 - p1
        
        # CP 로직
        pred_set = set()
        if s0 <= q0:
            pred_set.add(0)
        if s1 <= q1:
            pred_set.add(1)
        
        # CP 클래스 결정
        if len(pred_set) == 1:
            label = list(pred_set)[0]
            if label == 0:
                cp_class = 0
                cp_class_name = "명확한 정상"
            else:
                cp_class = 3
                cp_class_name = "명확한 포트홀"
        else:
            if p1 >= p0:
                cp_class = 2
                cp_class_name = "애매한 포트홀"
            else:
                cp_class = 1
                cp_class_name = "애매한 정상"
        
        # 단순 분류 (CP 클래스 기반)
        # 명확한 것만 선택, 애매한 것은 정상으로 취급
        if cp_class in [0, 3]:
            simple_class = "정상" if cp_class == 0 else "포트홀"
        else:
            simple_class = "정상"  # cp_class in [1, 2]: 애매하면 정상
        confidence = max(p0, p1)
        
        records.append({
            "프레임번호": frame_num,
            "타임스탐프_초": timestamp,
            "단순분류": simple_class,
            "정상확률": f"{p0:.4f}",
            "포트홀확률": f"{p1:.4f}",
            "신뢰도": f"{confidence:.4f}",
            "CP클래스": cp_class_name,
            "CP클래스번호": cp_class,
            "파일경로": str(frame_data['path']) if frame_data['path'] else "메모리"
        })
        
        if idx % 50 == 0 or idx == total:
            print(f"   {idx}/{total} 완료 ({idx/total*100:.1f}%)...")
    
    # DataFrame 생성
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Inference 완료!\n")
    
    return df


# ============================================================
# 결과 분석 및 시각화
# ============================================================
def analyze_results(df):
    """결과 분석 및 통계 출력"""
    
    print(f"{'='*80}")
    print(f"📊 결과 분석")
    print(f"{'='*80}\n")
    
    # CP 클래스 분포
    cp_counts = df["CP클래스"].value_counts()
    print(f"🏷️  CP 클래스 분포:")
    for cls, count in cp_counts.items():
        pct = count / len(df) * 100
        print(f"   {cls}: {count}개 ({pct:.1f}%)")
    
    # 단순 분류 분포
    simple_counts = df["단순분류"].value_counts()
    print(f"\n🏷️  단순 분류 분포:")
    for cls, count in simple_counts.items():
        pct = count / len(df) * 100
        print(f"   {cls}: {count}개 ({pct:.1f}%)")
    
    # 신뢰도 통계
    confidence = df["신뢰도"].astype(float)
    print(f"\n📈 신뢰도 통계:")
    print(f"   평균: {confidence.mean():.4f}")
    print(f"   최소: {confidence.min():.4f}")
    print(f"   최대: {confidence.max():.4f}")
    print(f"   중앙값: {confidence.median():.4f}")
    
    # Fake Log Generator를 위한 정보
    pothole_frames = df[df["단순분류"] == "포트홀"]
    print(f"\n💡 Fake Log Generator 설계용:")
    print(f"   포트홀 감지 프레임: {len(pothole_frames)}개")
    if len(pothole_frames) > 0:
        print(f"   포트홀 감지율: {len(pothole_frames)/len(df)*100:.1f}%")
        pothole_timestamps = pothole_frames["타임스탐프_초"].astype(float).tolist()
        print(f"   포트홀 발견 타임스탐프(초): {pothole_timestamps[:10]}{'...' if len(pothole_timestamps) > 10 else ''}")
    else:
        print(f"   포트홀 감지율: 0.0%")
        print(f"   포트홀 발견 타임스탐프(초): 없음")
    
    print(f"\n{'='*80}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="비디오 → 프레임 추출 & Inference 통합 파이프라인"
    )
    parser.add_argument("--video", type=str, required=True,
                       help="입력 비디오 파일 경로")
    # sample_maker 폴더 기준 경로
    script_dir = Path(__file__).parent
    parser.add_argument("--model", type=str, 
                       default=str(script_dir / "models" / "best.pt"),
                       help="모델 파일 경로")
    parser.add_argument("--cpjson", type=str, 
                       default=str(script_dir / "models" / "cp_results_90.json"),
                       help="CP results JSON 파일 경로")
    parser.add_argument("--frames_out", type=str, 
                       default=str(script_dir / "temp_frames"),
                       help="프레임 저장 폴더")
    parser.add_argument("--out", type=str, 
                       default=str(script_dir / "inference_results"),
                       help="결과 CSV 저장 폴더")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="사용할 장치 (cuda:0 또는 cpu)")
    parser.add_argument("--crop_bonnet", type=float, default=0.2,
                       help="본네트 crop 비율 (기본값: 0.2, 20 percent)")
    
    args = parser.parse_args()
    
    # 폴더 생성
    Path(args.frames_out).mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(args.video).stem
    output_csv = Path(args.out) / f"{video_name}_inference_{timestamp}.csv"
    
    try:
        # Step 1: 프레임 추출
        frames_info = extract_frames_from_video(args.video, args.frames_out)
        
        # Step 2: Inference 실행
        df = run_inference_on_frames(
            frames_info,
            args.model,
            args.cpjson,
            output_csv,
            device=args.device,
            crop_bonnet_ratio=args.crop_bonnet
        )
        
        # Step 3: 결과 분석
        analyze_results(df)
        
        # Step 4: 샘플 출력
        print(f"📋 샘플 결과 (처음 10개):")
        print(df[["프레임번호", "타임스탐프_초", "단순분류", "포트홀확률", "CP클래스"]].head(10).to_string(index=False))
        print()
        
        if len(df) > 10:
            print(f"📋 샘플 결과 (마지막 10개):")
            print(df[["프레임번호", "타임스탐프_초", "단순분류", "포트홀확률", "CP클래스"]].tail(10).to_string(index=False))
            print()
        
        print(f"{'='*80}")
        print(f"✅ 완료!")
        print(f"{'='*80}")
        print(f"📁 프레임 저장: {args.frames_out}")
        print(f"📊 결과 CSV: {output_csv}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()