import json

input_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/model_eval/_input/Lv4.jsonl"
output_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/model_eval/_input/Lv4.input_only.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        out = {
            "emotion": data.get("emotion", ""),
            "post_type": data.get("post_type", ""),
            "content": data.get("content", "")
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        