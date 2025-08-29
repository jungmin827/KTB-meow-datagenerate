import json
import os 

def convert_to_instruction_format(
    old_path: str,
    new_path: str
) -> None:
    with open(old_path, 'r', encoding='utf-8') as fin, open(new_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line)
            # 감정(emotion)이 영어라면 happy → happy한 등으로 변환
            instruction = f"다음 문장을 {data['post_type']}의 {data['emotion']}한 말투로 바꿔줘."
            new_data = {
                "instruction": instruction,
                "input": data["content"],
                "output": data["transformed_content"]
            }
            fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')

root_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/"

# 사용 예시
convert_to_instruction_format(
    os.path.join(root_path,"_made/dataset_0629_made.jsonl"),
    os.path.join(root_path, "_instruct/dataset_0629_instruct.jsonl")
)
# /Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_filtered/dataset_0619_filtered.jsonl