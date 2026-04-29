"""
sample_maker용 간단 추론 스크립트

사용법:
    python inference_simple.py --input ./test_images
"""

import os
import argparse
import json
from pathlib import Path
import shutil
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

from config import create_model  # 모델 생성 함수


# ============================================================
# 이미지 불러오기
# ============================================================
def load_images_from_folder(folder, imgsz=224):
    folder = Path(folder)
    exts = ["jpg", "png", "jpeg", "bmp", "webp"]
    files = []
    for ext in exts:
        files += list(folder.rglob(f"*.{ext}"))
    return sorted(files)


# ============================================================
# 전처리 함수
# ============================================================
def preprocess(img_path, imgsz=224):
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


# ============================================================
# CP Inference Main
# ============================================================
def run_cp_inference(input_folder, model_path, cp_json_path, output_root):

    folder_name = Path(input_folder).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_csv = Path(output_root) / f"{folder_name}_results_{timestamp}.csv"
    
    print(f"\n{'='*80}")
    print(f"🔍 이미지 분석 시작")
    print(f"{'='*80}")
    print(f"입력: {input_folder}")
    print(f"모델: {model_path}")
    print(f"CP JSON: {cp_json_path}")
    print(f"{'='*80}\n")
    
    # -----------------------------
    # CP JSON 로드
    # -----------------------------
    with open(cp_json_path, "r") as f:
        cp_data = json.load(f)

    quantiles = cp_data["cp_results"]["quantiles"]
    q0 = float(quantiles["0"])
    q1 = float(quantiles["1"])
    
    print(f"✅ CP Quantiles")
    print(f"   q0 (Normal):  {q0:.4f}")
    print(f"   q1 (Pothole): {q1:.4f}\n")

    # -----------------------------
    # 모델 로드
    # -----------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _ = create_model("last_4_blocks", device)

    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.model.load_state_dict(ckpt)

    print(f"✅ 모델 로드 완료")
    print(f"   장치: {device}\n")

    # -----------------------------
    # 이미지 불러오기
    # -----------------------------
    images = load_images_from_folder(input_folder)
    print(f"📊 총 {len(images)}장 발견\n")

    if len(images) == 0:
        print("❌ 이미지가 없습니다!")
        return

    records = []

    # -----------------------------
    # CP Logic
    # -----------------------------
    print("🔍 분석 중...\n")
    
    for idx, img_path in enumerate(images, 1):

        filename = Path(img_path).name

        img_tensor = preprocess(img_path).to(device)

        with torch.no_grad():
            logits = model.model(img_tensor)
            logits = logits[0] if isinstance(logits, tuple) else logits
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

        p0, p1 = float(prob[0]), float(prob[1])
        s0, s1 = 1 - p0, 1 - p1

        pred_set = set()
        if s0 <= q0:
            pred_set.add(0)
        if s1 <= q1:
            pred_set.add(1)

        # CP 클래스 결정
        if len(pred_set) == 1:
            label = list(pred_set)[0]
            if label == 0:
                cp_class = 0    # 확실한 정상
                cp_class_name = "확실한 정상"
            else:
                cp_class = 3    # 확실한 포트홀
                cp_class_name = "확실한 포트홀"
        else:
            if p1 >= p0:
                cp_class = 2  # 애매 포트홀
                cp_class_name = "애매 (포트홀 쪽)"
            else:
                cp_class = 1  # 애매 정상
                cp_class_name = "애매 (정상 쪽)"

        # 단순 분류
        simple_class = "포트홀" if p1 > p0 else "정상"
        confidence = max(p0, p1)

        records.append({
            "번호": idx,
            "파일명": filename,
            "단순분류": simple_class,
            "정상확률": f"{p0:.4f}",
            "포트홀확률": f"{p1:.4f}",
            "신뢰도": f"{confidence:.4f}",
            "CP예측집합": str(pred_set),
            "CP클래스": cp_class_name,
            "CP클래스번호": cp_class,
            "이미지경로": str(img_path)
        })
        
        if idx % 100 == 0 or idx == len(images):
            print(f"   {idx}/{len(images)} 완료...")

    # -----------------------------
    # CSV 저장
    # -----------------------------
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ 완료!\n")
    
    # -----------------------------
    # 요약 통계
    # -----------------------------
    print(f"{'='*80}")
    print(f"📊 결과 요약")
    print(f"{'='*80}\n")
    
    cp_counts = df["CP클래스"].value_counts()
    print(f"CP 클래스 분포:")
    for cls, count in cp_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count}장 ({pct:.1f}%)")
    
    print(f"\n평균 신뢰도: {df['신뢰도'].astype(float).mean():.4f}")
    
    print(f"\n💾 결과 저장:")
    print(f"   {output_csv}")
    print(f"{'='*80}\n")
    
    # 샘플 출력
    print("📋 샘플 결과 (처음 5개):")
    print(df.head()[["파일명", "단순분류", "정상확률", "포트홀확률", "CP클래스"]].to_string(index=False))
    print()


# ============================================================
# Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    
    # sample_maker 기준 경로
    script_dir = Path(__file__).parent
    parser.add_argument("--model", type=str,
                        default=str(script_dir / "models" / "best.pt"))
    parser.add_argument("--cpjson", type=str,
                        default=str(script_dir / "models" / "cp_results_90.json"))
    parser.add_argument("--out", type=str,
                        default=str(script_dir / "inference_results"))

    args = parser.parse_args()
    
    # 출력 폴더 생성
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    run_cp_inference(args.input, args.model, args.cpjson, args.out)


if __name__ == "__main__":
    main()