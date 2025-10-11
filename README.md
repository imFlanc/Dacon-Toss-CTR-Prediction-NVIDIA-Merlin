# Dacon-Toss-CTR-Prediction-NVIDIA-Merlin

# 📢 CTR Prediction with NVIDIA Merlin & XGBoost

본 프로젝트에서는 **NVIDIA Merlin + RAPIDS** 생태계 기반으로 대규모 **CTR 예측**을 수행합니다.  
GPU 가속 전처리(NVTabular)와 **XGBoost 모델 학습**, 간단한 weighted **이진 로그 손실(Binary Log Loss)** 기반 추론을 포함합니다.

---

## 📁 포함 파일

- `train.ipynb` : 데이터 로딩, 전처리, 학습 전체 파이프라인
- `inference.ipynb` : 추론 전체 파이프라인
- `Dockerfile` : CUDA 11.8 기반 Merlin + RAPIDS 실행 환경 (NVIDIA RTX A6000 Workstation)
- `requirements.txt` : Python Core + ML/Visualization + XGBoost 패키지 목록

---

## 📊 데이터 요약

- Train: 10,704,179 rows
- Test: 1,527,298 rows
- Feature: 119 cols

---

## 🚀 실행 환경

### Docker로 실행
```bash
docker run --gpus all -it --ipc=host -p 8888:8888 \
  -v ~/Toss-Dacon:/workspace \
  --name toss-container \
  toss-merlin-env \
  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04  
