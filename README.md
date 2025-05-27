# SNS 원문 동물 변환 프로젝트

이 프로젝트는 SNS 원문 텍스트를 고양이와 강아지의 다양한 감정으로 변환하는 프로그램입니다.

## 프로젝트 설명

- 원본 JSON 파일에 있는 각 텍스트를 읽어옵니다.
- 각 원문마다 무작위로 2개의 동물-감정 조합을 선택합니다 (예: 고양이-행복, 강아지-슬픔).
- Google Gemini API를 사용하여 원문을 선택된 동물과 감정 스타일로 변환합니다.
- 변환된 결과를 JSONL 형식으로 저장합니다.

## 필요 라이브러리

```bash
pip install google-generativeai python-dotenv
```

## 사용 방법

1. `.env` 파일에 Gemini API 키를 설정합니다:
   ```
   GOOGLE_API_KEY=여기에_API_키_입력
   ```

2. 스크립트 실행:
   ```bash
   python transform_content.py --input_file 데이터/0526sns_content.json --output_file 데이터/transformed_content.jsonl
   ```

### 명령행 인자

- `--input_file`: 입력 JSON 파일 경로 (기본값: '데이터/0526sns_content.json')
- `--output_file`: 출력 JSONL 파일 경로 (기본값: '데이터/transformed_content.jsonl')
- `--api_key`: Google Gemini API 키 (환경 변수 GOOGLE_API_KEY가 설정되어 있지 않은 경우에만 필요)
- `--target_count`: 목표 생성 항목 수 (기본값: 564)

## 출력 파일 형식

출력 파일은 JSONL 형식으로, 각 줄은 다음과 같은 JSON 객체입니다:

```json
{
  "content": "원본 텍스트",
  "emotion": "감정 (happy, normal, grumpy, angry, curious, sad 중 하나)",
  "post_type": "동물 타입 (cat 또는 dog)",
  "transformed_content": "변환된 텍스트"
}
```

## 감정 종류

- happy: 행복한
- normal: 일반적인
- grumpy: 까칠한
- angry: 화난
- curious: 호기심 많은
- sad: 슬픈

## 참고

- normal과 happy 감정이 약간 더 많이 생성되도록 가중치를 설정했습니다.
- 각 원문마다 서로 다른 두 개의 동물-감정 조합이 적용됩니다. 