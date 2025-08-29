import json

def filter_jsonl_bytes_by_threshold(
    eval_jsonl_bytes_list,
    thresholds: dict
):
    filtered = []
    for eval_bytes in eval_jsonl_bytes_list:
        lines = eval_bytes.decode("utf-8").splitlines()
        for line in lines:
            data = json.loads(line)
            passed = True
            for key, thres in thresholds.items():
                value = data.get(key)
                if value is None or float(value) < thres:
                    passed = False
                    break
            if passed:
                filtered.append(data)
    return filtered

# def filter_normal_kobertscore(data_list, kobertscore_threshold=0.6):
#     """
#     data_list: list of dict (이미 ''json.loads 된 상태)
#     kobertscore_threshold: float (이 값 이하인 normal 감정 row 제거)
#     """
#     result = []
#     for data in data_list:
#         if data.get("emotion") == "normal":
#             kobertscore = data.get("kobertscore")
#             if kobertscore is not None and float(kobertscore) <= kobertscore_threshold:
#                 continue  # 조건에 해당하면 건너뜀
#         result.append(data)
#     return result
