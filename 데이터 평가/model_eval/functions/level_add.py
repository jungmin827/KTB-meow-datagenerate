import json

def assign_level(entry, level: str):
    """
    level 값을 그대로 문자열로 저장 (ex: '1', '2', '3', '4')
    """
    return level

def add_level_to_jsonl(input_path: str, output_path: str, level: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry["level"] = assign_level(entry, level)
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")

# ✅ 여기서 직접 설정
if __name__ == "__main__":
    input_jsonl_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/model_eval/_input/Lv4.input_only.jsonl"          # 원본 JSONL 경로
    output_jsonl_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/model_eval/_input/_input_lv.jsonl"    # 출력 JSONL 경로
    level = "4"                               # level 값 (문자열 숫자)

    add_level_to_jsonl(input_jsonl_path, output_jsonl_path, level)
    print(f"✅ 완료: '{output_jsonl_path}'에 level='{level}'이 추가되었습니다.")