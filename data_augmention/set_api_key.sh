#!/bin/bash

# API 키를 환경 변수로 설정하는 스크립트

# 사용 방법 안내
if [ "$#" -ne 1 ]; then
    echo "사용법: source set_api_key.sh YOUR_API_KEY"
    echo "예시: source set_api_key.sh abcd1234efgh5678"
    exit 1
fi

# API 키 환경 변수 설정
export GEMINI_API_KEY="$1"
echo "GEMINI_API_KEY 환경 변수가 설정되었습니다."
echo "이제 다음과 같이 스크립트를 실행할 수 있습니다:"
echo "python cat_dog_filter.py --input_file 0523captions_only.json --output_file filtered_cat_dog.json --api_key \$GEMINI_API_KEY --num_samples 300" 