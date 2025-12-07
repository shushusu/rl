# ESS Control with Hierarchical MARL

Transformer 기반 Hierarchical Multi-Agent Reinforcement Learning(H-MARL)로
Energe Storage System(ESS)의 충·방전을 최적화하는 프로젝트입니다.

## 1. 프로젝트 구조
- **preprocessing.py** — CSV → numpy 전처리
- **hierarchical_env.py** — Manager/Worker 환경 정의
- **train_hierarchical.py** — Transformer PPO 학습
- **eval_a1_experiment.py** — A1 baseline 비교 (No-ESS / Rule / H-Trans)
- **eval_a2_experiment.py** — A2 reward tuning 평가
- **eval_a2_seed.py** — Multi-seed 성능 검증
- **ma_train_data.npy** — 전처리된 학습 데이터

## 2. 환경 설정
pip install numpy pandas matplotlib 
pip install torch

## 3. 데이터 준비
Kaggle Plant 2 Dataset 필요

Plant_2_Generation_Data.csv

Plant_2_Weather_Sensor_Data.csv

전처리 실행
python preprocessing.py


실행 결과

ma_train_data.npy 생성

## 3. 데이터 준비

본 프로젝트는 **Kaggle Plant 2 Dataset**을 사용합니다.

필요한 원본 CSV 파일:
- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

### ▶ 전처리 실행
```bash
python preprocessing.py

전처리 결과 생성 파일:

ma_train_data.npy


## 4. 학습 실행

Transformer 기반 Manager/Worker H-MARL 학습

python train_hierarchical.py


출력

manager_transformer_*.pth

worker_transformer_*.pth


## 5. 평가 실행
A1 실험: Baseline 비교 (No ESS vs Rule vs H-Trans)
python eval_a1_experiment.py

A2 실험: Reward Tuning 영향 평가
python eval_a2_experiment.py

Multi-Seed 실험 
python eval_a2_seed.py

## 6. 주요 결과 요약
A1 Baseline 비교

Rule-based ESS가 No-ESS보다 비용 감소

초기 버전 H-Trans는 reward 설계의 한계로 개선 필요

A2 Reward Tuning 후

H-Trans가 비용·피크·안정성 모두 향상

Worker/Manager 협업 구조가 제대로 작동하기 시작함

Multi-Seed 실험 결과

서로 다른 seed(0/1/2)에서도 일관된 학습 패턴

학습 곡선이 동일한 형태로 수렴 
