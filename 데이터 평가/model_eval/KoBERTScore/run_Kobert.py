import json
import sys
import argparse
import getenv

sys.path.append("/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/KoBERTScore")

from KoBERTScore.score import BERTScore

model_name = "beomi/kcbert-base"
bertscore = BERTScore(model_name, best_layer=4)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

dataset = args.dataset
input_jsonl = f"/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/{dataset}.jsonl"
output_jsonl = f"/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/{dataset}_kbs.jsonl"

data_list = []
candidates = []
references = []
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
# 토큰 수를 기반으로 300개에서 자르는 코드 작성
def truncate_text(text: str, max_tokens: int = 290) -> str:
    # 토큰 ID로 변환
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    # 300개 초과 시 자르기
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]
    # 다시 문자열로 복원
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# 계산 진행
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "instruct" in dataset:
            candidates.append(truncate_text(data["input"]))
            references.append(truncate_text(data["output"]))
        else: 
            candidates.append(truncate_text(data["content"]))
            references.append(truncate_text(data["transformed_content"]))
        data_list.append(data)

print(f"⭕ 총 {len(data_list)}개 데이터, BERTScore 계산 시작...")
print(f"-> 데이터셋 : {dataset}")

scores = bertscore(candidates, references, batch_size=128)

bar_length = 40
for idx, (data, f1) in enumerate(zip(data_list, scores), 1):
    data["kobertscore_f1"] = float(f1)
    if idx % 1 == 0 or idx == len(data_list):
        percent = idx / len(data_list)
        filled_len = int(bar_length * percent)
        bar = "|" * filled_len + "-" * (bar_length - filled_len)
        print(f"\r진행률: |{bar}| {idx}/{len(data_list)} ({percent*100:.1f}%)", end="")
        sys.stdout.flush()

print() 

with open(output_jsonl, "w", encoding="utf-8") as f:
    for data in data_list:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ kobertscore_f1 계산이 완료 되었습니다.")
print(f"-> 결과 파일: {output_jsonl}")