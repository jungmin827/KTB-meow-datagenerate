import json

input_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_made/dataset_0629_made.jsonl"     # 원본 파일명 (네 파일명에 맞게)
output_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_made/dataset_0709_made.json" # 출력 파일명

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        if data.get("emotion") != "normal":
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")