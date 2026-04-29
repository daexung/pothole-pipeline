# Pothole Pipeline

> AI 기반 도로 유지보수 의사결정을 위한 이벤트 중심 데이터 파이프라인 프로젝트

최근 지자체 및 대중교통 차량에는 AI 연산이 가능한 Edge Device 기반 스마트 블랙박스 도입이 확대되고 있습니다.
본 프로젝트는 이러한 차량 환경을 가정하여, **버스 주행 중 발생하는 포트홀 탐지 이벤트를 실시간 수집·적재·분석하는 데이터 파이프라인 구축**을 목표로 합니다.

실제 상용 AI 블랙박스 장비 확보가 어려워, 로컬 PC 또는 Amazon Web Services Amazon EC2 환경을 가상의 Edge Device로 구성했습니다.
단일 샘플 영상을 입력 스트림으로 활용하여 프레임 샘플링, 모델 추론, 이벤트 JSON 생성 과정을 시뮬레이션했습니다.

또한 수백 대 차량이 동시에 운행되는 대규모 운영 환경을 가정하여, 추가 차량·노선 데이터는 Faker 기반 실시간 로그 생성기로 대체했습니다. 이를 통해 **다중 차량 이벤트 수집 구조와 ETL 파이프라인 확장성**을 검증합니다.

---

## Project Goals

* 모델 성능 개선보다 **운영 가능한 데이터 파이프라인 설계**에 집중
* 포트홀 탐지 이벤트를 JSON 형태로 표준화
* 위치 / 노선 / 회차 / Confidence 기반 분석 구조 설계
* Apache Airflow 기반 ETL 자동화
* PostgreSQL 적재 및 조회 구조 구축
* FastAPI API 제공
* Dashboard / 지도 시각화 환경 구축
* Edge Device 기반 실시간 이벤트 수집 구조 시뮬레이션

---

## Architecture

```text
[Vehicle Edge Device / AI Blackbox]
Video Stream
→ Frame Sampling
→ ROI Crop / Resize
→ Model Inference
→ Event JSON 생성

[Streaming / Storage Layer]
Event Logs
→ S3 / PostgreSQL 적재

[Batch Pipeline]
Airflow ETL
→ 정제 / 집계 / 적재

[Serving Layer]
FastAPI
→ Dashboard / Map Visualization
```

---

## Core Features

### 1. Edge Device 실시간 추론 시뮬레이션

* 샘플 블랙박스 영상을 실시간 스트림처럼 처리
* 프레임 샘플링 후 ROI 전처리 수행
* 포트홀 탐지 이벤트 발생 시 JSON 로그 생성

### 2. 대규모 차량 로그 생성

* Faker 기반 버스 / 노선 / GPS / 운행 데이터 생성
* 수백 대 차량 동시 운행 환경 시뮬레이션

### 3. ETL Pipeline

* 이벤트 로그 수집
* 데이터 정제 및 중복 제거
* 통계 집계 및 DB 적재

### 4. 서비스 활용 예시

* 포트홀 다발 지역 시각화
* 노선별 도로 상태 분석
* 우선 보수 지역 선정
* 민원 발생 전 선제 유지보수 의사결정 지원

---

## Tech Stack

* Python
* OpenCV
* YOLOv8-cls
* Faker
* PostgreSQL
* Apache Airflow
* FastAPI
* Docker
* AWS S3 / EC2

---

## Dataset Sources

### Dashcam Images

Kaggle
100K Vehicle Dashcam Image Dataset

### Pothole Images

AIHub
포트홀 데이터셋

---

## Expected Business Impact

* 도로 유지보수 비용 절감
* 민원 대응 이전 선제적 조치 가능
* 차량 주행 데이터를 활용한 스마트 인프라 운영
* 지자체 / 건설사 데이터 기반 의사결정 지원

---

## Disclaimer

본 프로젝트는 포트폴리오 및 학습 목적의 시뮬레이션 프로젝트이며, 공개 데이터셋을 활용해 재구성되었습니다.


## sample_maker 사용법

pip install torch torchvision ultralytics opencv-python pandas pillow

pip install opencv-python numpy