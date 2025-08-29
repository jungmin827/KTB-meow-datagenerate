#!/bin/bash

# (1) 기존 컨테이너 중지 및 삭제 (있을 경우)
docker stop model-eval-app-container 2>/dev/null
docker rm model-eval-app-container 2>/dev/null

# (2) 이미지 빌드
docker build -t model-eval-app .

# (3) 컨테이너 실행 (포트 7860 매핑, 이름 지정)
docker run --name model-eval-app-container -p 7860:7860 model-eval-app