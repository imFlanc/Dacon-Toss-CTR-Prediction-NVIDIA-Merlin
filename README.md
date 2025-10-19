# 📢 CTR Prediction with NVIDIA Merlin & XGBoost

본 프로젝트에서는 GPU 가속 전처리 도구인 **NVTabular**와,  
**NVIDIA Merlin + RAPIDS 생태계**를 활용해 대규모 **CTR 예측(Click-Through Rate Prediction)** 을 수행합니다.  

NVIDIA RTX A6000 워크스테이션 환경에서 전체 데이터셋(10.7M)을 대상으로  
Stratified 5-Fold Cross-Validation을 수행할 경우, fold당 약 30초, 전체 학습에 약 2분이 소요됩니다.  
- NVIDIA Merlin 공식 문서: https://developer.nvidia.com/merlin

학습에는 **XGBoost GPU predictor** 를 사용하며,  
positive/negative 클래스 불균형을 보정하기 위해 **클래스 가중치 이진 로그 손실 (class-weighted binary log loss)** 을 적용합니다.

추론은 **DMatrix 기반 GPU predictor**를 통해 수행되며, 테스트 데이터셋(1.5M)에 대해 약 140초가 소요됩니다.  

학습 및 추론 과정은 GPU 메모리 사용 효율을 고려하여 NVTabular의 스트리밍 기반 파이프라인으로 구성되어 있습니다.  

---

## 📁 포함 파일

- `train.ipynb` : 데이터 로딩, 전처리, 학습 전체 파이프라인 
- `inference.ipynb` : 추론 전체 파이프라인 
- `Dockerfile` : CUDA 11.8 기반 Merlin + RAPIDS 실행 환경 
- `requirements.txt` : Python Core + ML/Visualization + XGBoost 패키지 목록
- `to_oonx.ipynb` : **프레임워크 독립적 배포(IR)** 형태의 모델 추론 파이프라인 구축용 모델 형식 변환  
*(ONNXRuntime 및 TensorRT 변환 테스트용)*

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
