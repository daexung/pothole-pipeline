# Pothole Pipeline

대중교통 차량의 블랙박스 영상 데이터를 수집·전처리·적재한 뒤,
AI 기반 포트홀 탐지 모델을 통해 도로 파손 여부를 판별하고,
위치 기반 대시보드에서 유지보수 우선순위를 확인할 수 있도록 설계한 데이터 파이프라인 프로젝트입니다.

## 프로젝트 목표

- 모델 성능 개선보다 파이프라인 구축 및 배포 가능한 구조 설계에 집중
- 포트홀 탐지 결과 데이터를 정형화하여 PostgreSQL에 적재
- 노선 / 회차 / 위치 / 신뢰도 기반 분석 구조 설계
- Airflow 기반 ETL 파이프라인 자동화
- FastAPI 또는 대시보드를 통한 결과 조회 및 시각화

## Architecture

```text
Blackbox Video Data
→ Frame Extraction
→ Preprocessing
→ Model Inference
→ PostgreSQL
→ API / Dashboard