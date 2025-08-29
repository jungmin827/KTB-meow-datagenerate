# KoBERTScore 사용법

## 1. 설치 방법

```bash
pip install git+https://github.com/lovit/KoBERTScore.git
cd ko-BERTScore
python setup.py install
```

---

## 2. 실행 방법

- **실행 파일:**  
  `run.py`  
  (예시 경로: `for_AI/my_project/Finetuning/dataset/KoBERTScore/run.py`)

- **실행 명령어:**  
  ```bash
  python run.py
  ```

---

## 3. 입력/출력 파일

- **입력 파일:**  
  `/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/dataset_0615_instruct.jsonl`  
  (각 줄마다 JSON 객체, 예: `{"input": "...", "output": "..."}`)

- **출력 파일:**  
  `/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/dataset_0615_instruct_with_kobertscore.jsonl`  
  (각 데이터에 `"kobertscore_f1"` 필드가 추가됨)

---

## 4. 실행 결과 예시

- 각 데이터에 `"kobertscore_f1"` 값이 추가되어 저장됩니다.
- 예시:
  ```json
  {
    "instruction": "다음 문장을 dog의 normal한 말투로 바꿔줘.",
    "input": "나는 댕댕이일까 냐용이일까요~",
    "output": "나는 댕댕이일까 냐용이일까, 그것이 궁금하다멍! 꼬리 흔들고 킁킁 냄새 맡으면 알 수 있을까 왈? 🐾",
    "kobertscore_f1": 0.7726503014564514
  }
  ```

---

## 5. KoBERTScore 해석

- `kobertscore_f1` 값은 **0~1 사이**의 실수입니다.
- 1에 가까울수록 정답(output)과 생성문장이 의미적으로 유사함을 의미합니다.
- 여러 샘플의 평균을 내어 전체 성능을 평가할 수 있습니다.

---

## 6. 참고

- 진행상황 및 결과 파일 경로는 실행 시 터미널에 출력됩니다.
- 입력/출력 파일 경로는 `run.py` 코드에서 직접 수정할 수 있습니다.