# 📢 CTR Prediction with NVIDIA Merlin & XGBoost

본 프로젝트에서는 GPU 가속 전처리(NVTabular)와 **NVIDIA Merlin + RAPIDS** 생태계 기반으로 대규모 **CTR 예측**을 수행합니다.  
NVIDIA RTX A6000 Workstation 환경에서 전체 데이터(10.7M) 5fold-CV(stratifiedKfold) 학습에 약 2분 가량이 소요됩니다 (30sec/folds).  
(https://developer.nvidia.com/merlin)

학습에는 **XGBoost 모델**을 사용하였고,  
positive/negative 클래스 불균형을 보정하기 위해 간단한 class weighted **이진 로그 손실(Binary Log Loss)** 을 사용합니다:  
<p align="center">
  <img src="https://github.com/user-attachments/assets/ba9692c7-fc1f-409f-be00-31f0de60c9a8" width="192" height="128" alt="weighted loss formula" />
</p>

---

## 📁 포함 파일

- `train.ipynb` : 데이터 로딩, 전처리, 학습 전체 파이프라인 
- `inference.ipynb` : 추론 전체 파이프라인 
- `Dockerfile` : CUDA 11.8 기반 Merlin + RAPIDS 실행 환경 
- `requirements.txt` : Python Core + ML/Visualization + XGBoost 패키지 목록 

---

## 📊 데이터 요약

- Train: 10,704,179 rows
- Test: 1,527,298 rows
- Feature: 119 cols (anonymized)

---

## 🚀 실행 환경

### Docker
```bash
docker run --gpus all -it --ipc=host -p 8888:8888 \
  -v ~/Toss-Dacon:/workspace \
  --name toss-container \
  toss-merlin-env \
  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''
```  
### Dockerfile (Base Image)
```
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04  
```

---
### 📌 Acknowledgement  
Parts of the training and inference code are derived from community-shared solutions in the  [Dacon CTR Prediction Competition](https://dacon.io/competitions/official/236575/overview/description).  
We have modified the code to suit our custom hardware and runtime environment, including support for NVIDIA Merlin and RAPIDS-based GPU pipelines.

Additional inspiration and implementation details were adapted from the official NVIDIA Merlin examples:  
https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/Building-and-deploying-multi-stage-RecSys
