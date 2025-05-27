#!/bin/bash

# 고양이-강아지 문장 변환 데이터 필터링 실행 예제 스크립트

# API 키를 설정합니다 (실제 사용 시 본인의 API 키로 변경하세요)
API_KEY="AIzaSyD23XRI2GJiJR6coDBnb5GcIVN9NasJjtY"

# 기본 필터링 (5개 샘플만 처리)
echo "기본 필터링 실행 중..."
python cat_dog_filter.py --input_file 0523captions_only.json --output_file filtered_cat_dog_simple.json --api_key $API_KEY --num_samples 5

# 결과 확인
echo "기본 필터링 결과 확인:"
head -n 20 filtered_cat_dog_simple.json

# 배치 처리 (10개 샘플, 배치 크기 5, 병렬 워커 2개로 처리)
echo "배치 처리 실행 중..."
python batch_processor.py --input_file 0523captions_only.json --output_file filtered_cat_dog_batch.json --api_key $API_KEY --num_samples 10 --batch_size 5 --max_workers 2 --checkpoint_interval 1

# 결과 확인
echo "배치 처리 결과 확인:"
head -n 20 filtered_cat_dog_batch.json

# 변환 예시 확인
echo "변환 예시 확인:"
head -n 20 transformation_examples.json

echo "스크립트 실행 완료!" 