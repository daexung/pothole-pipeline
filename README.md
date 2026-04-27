# Pothole Pipeline

지자체 및 대중교통 차량에는 AI 기반 블랙박스(AI 연산이 가능한 Edge Device)가 점차 도입되고 있다.

본 프로젝트는 이러한 차량용 AI 블랙박스 환경을 가정하여, 버스 주행 중 발생하는 포트홀 탐지 이벤트를 실시간 수집·적재·분석하는 데이터 파이프라인 구축을 목표로 한다.

실제 상용 장비를 확보하기 어려워 로컬 PC 또는 EC2 환경을 가상의 블랙박스로 설정하였으며, 단일 샘플 영상 데이터를 활용해 프레임 샘플링, 모델 추론, 이벤트 JSON 생성 과정을 시뮬레이션하였다.

또한 대규모 운영 환경을 가정하여, 추가 차량·노선 데이터는 Faker 기반 실시간 로그 생성기로 대체함으로써 다수 차량에서 발생하는 이벤트 수집 및 ETL 파이프라인 확장성을 검증한다.

## Project Goals

- 모델 성능 개선보다 운영 가능한 데이터 파이프라인 설계에 집중
- 포트홀 탐지 이벤트를 JSON 형태로 표준화하고 PostgreSQL에 적재
- 노선 / 회차 / 위치 / Confidence 기반 분석 구조 설계
- Airflow 기반 ETL 스케줄 자동화
- FastAPI 및 Dashboard 기반 조회 / 시각화 환경 구축
- Edge Device 기반 이벤트 수집 구조 시뮬레이션

## Architecture

```text
[Edge Device / AI Blackbox]
Sample Video Stream
→ Frame Sampling
→ Preprocessing (ROI / Resize)
→ Model Inference
→ Event JSON 생성

[Cloud Pipeline]
Event Logs
→ S3 / PostgreSQL 적재
→ Airflow ETL
→ FastAPI
→ Dashboard / Map Visualization