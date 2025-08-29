import json

input_file = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_made/posts_dump_0709.json"    # 원본 JSON 파일명 (배열 형태)
output_file = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_made/posts_dump_0709.jsonl" # 변환될 JSONL 파일명

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 리스트로 로드됨

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
