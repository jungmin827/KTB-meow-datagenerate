import json
import os
from collections import OrderedDict

# 데이터 파일 경로 설정
root_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset"
input_name = "data_nyang.jsonl"
output_name = "dataset_0613_made.jsonl"
input_path = os.path.join(root_path, input_name)
output_path = os.path.join(root_path, output_name)

# 컬럼명 변경 매핑: 원본 컬럼명 → 바꿀 컬럼명
rename_map = {
    "input": "content",
    "output" : "transformed_content",
}

# 최종적으로 저장할 컬럼 순서 지정
column_order = ["content", "emotion", "post_type", "transformed_content"]

with open(input_path, encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue  # 빈 줄은 건너뜀

        # 한 줄씩 JSON 파싱
        item = json.loads(line)

        # 컬럼명 변경 적용
        new_item = {}
        for k, v in item.items():
            new_key = rename_map.get(k, k)  # 매핑에 있으면 바꿔주고, 없으면 그대로
            new_item[new_key] = v

        # emotion, post_type 컬럼 추가/수정
        new_item["emotion"] = "normal"
        new_item["post_type"] = "cat"

        # 지정한 컬럼 순서대로 정렬
        ordered = OrderedDict()
        for col in column_order:
            if col in new_item:
                ordered[col] = new_item[col]

        # 지정한 컬럼 외 나머지 컬럼도 뒤에 추가
        for k, v in new_item.items():
            if k not in ordered:
                ordered[k] = v

        # 한 줄씩 JSONL로 저장
        json.dump(ordered, outfile, ensure_ascii=False)
        outfile.write('\n')