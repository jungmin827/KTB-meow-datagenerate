#!/bin/bash

# 환경 변수를 활용하여 고양이-강아지 문장 변환 데이터 필터링 실행 스크립트

# API 키 환경 변수 확인
if [ -z "$GEMINI_API_KEY" ]; then
    echo "오류: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다."
    echo "다음 명령을 실행하여 API 키를 설정해주세요:"
    echo "source set_api_key.sh YOUR_API_KEY"
    exit 1
fi

# 필터링할 샘플 수 설정 (기본값: 300)
NUM_SAMPLES=${1:-300}

# 최대 문장 길이 설정 (기본값: 200)
MAX_LENGTH=${2:-200}

echo "고양이-강아지 문장 변환 데이터 필터링을 시작합니다."
echo "처리할 샘플 수: $NUM_SAMPLES"
echo "최대 문장 길이: $MAX_LENGTH자"

# 배치 처리 실행
echo "배치 처리 실행 중..."
python batch_processor.py \
    --input_file 0524captions_only2.json \
    --output_file filtered_cat_dog_2.json \
    --api_key $GEMINI_API_KEY \
    --num_samples $NUM_SAMPLES \
    --max_length $MAX_LENGTH \
    --batch_size 10 \
    --max_workers 5 \
    --checkpoint_interval 10

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo "필터링이 성공적으로 완료되었습니다!"
    echo "결과 파일: filtered_cat_dog_${NUM_SAMPLES}.json"
    echo "변환 예시 파일: transformation_examples.json"
else
    echo "필터링 중 오류가 발생했습니다."
    echo "로그 파일을 확인해보세요: batch_process.log"
fi 