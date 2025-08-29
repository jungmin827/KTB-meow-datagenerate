import json
from collections import Counter, defaultdict

def get_data_distribution(jsonl_bytes) -> dict:
    """
    업로드된 jsonl 파일(바이트)을 받아 데이터 분포 통계 반환
    """
    emotion_order = ["normal", "happy", "sad", "grumpy", "angry", "curious"]
    type_emotion_counter = defaultdict(lambda: Counter())
    type_total_counter = Counter()
    content_set = set()
    total_count = 0

    for line in jsonl_bytes.splitlines():
        line = line.decode("utf-8").strip()
        if not line:
            continue
        data = json.loads(line)
        total_count += 1
        post_type = data.get('post_type', 'unknown')
        emotion = data.get('emotion', 'unknown')
        content = data.get('content', '').strip()
        if content:
            content_set.add(content)
        type_emotion_counter[post_type][emotion] += 1
        type_total_counter[post_type] += 1

    return {
        "total_count": total_count,
        "unique_content_count": len(content_set),
        "type_emotion_counter": type_emotion_counter,
        "type_total_counter": type_total_counter,
        "emotion_order": emotion_order
    }