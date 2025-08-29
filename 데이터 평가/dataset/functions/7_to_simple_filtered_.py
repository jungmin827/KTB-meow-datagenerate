import re
import json

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return text

    # 1. 줄바꿈/탭 → 공백, 연속 공백 정리
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # 2. 깨진 URL 감지 및 합치기 (마침표 보존, 공백만 제거)
    def fix_url(m):
        url = m.group(1)
        # URL 내부 공백만 제거, 마침표 등은 보존
        url_clean = re.sub(r'\s+', '', url)
        return f'[{url_clean}]'
    # URL: https로 시작, 영어/숫자/특수문자(-._~:/?#[]@!$&'()*+,;=)만 포함, 한글/공백/이모지에서 종료
    url_broken_pattern = re.compile(
        r'(https?://[A-Za-z0-9\-\._~:/\?#\[\]@!\$&\'\(\)\*\+,;=%]+)'
    )
    text = url_broken_pattern.sub(fix_url, text)

    # 3. 구두점 앞에 붙은 공백 제거
    text = re.sub(r'\s+([?.!])', r'\1', text)

    # 4. 마침표(.) 뒤에 '한글'이 나오면 공백 추가 (영어나 숫자는 영향 X)
    text = re.sub(r'(\.)([가-힣])', r'\1 \2', text)

    # 5. 기타 전처리
    text = re.sub(r'^[\s.,?!·~…]+|[\s.,?!·~…]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
            except Exception:
                continue
            for key in ['content', 'transformed_content']:
                if key in data and isinstance(data[key], str):
                    data[key] = preprocess(data[key])
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 파일 경로는 사용자 환경에 맞게 수정
    in_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/_final/dataset_0710_all.jsonl"
    out_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/_final/dataset_0710_all_pre.jsonl"
    preprocess_jsonl(in_path, out_path)
    print(f"✅ 전처리 완료: {out_path}")
