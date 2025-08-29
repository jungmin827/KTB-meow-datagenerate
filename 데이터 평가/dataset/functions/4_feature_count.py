import json
import argparse
from collections import Counter, defaultdict

def count_posttype_emotion_and_content(input_path: str) -> None:
    """
    post_type별로 emotion 분포, 총 데이터 개수, content 중복 없는 개수 출력
    """
    emotion_order = ["normal", "happy", "sad", "grumpy", "angry", "curious"]
    type_emotion_counter = defaultdict(lambda: Counter())
    type_total_counter = Counter()
    content_set = set()
    total_count = 0

    with open(input_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
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

    print(f"\n✅ 총 데이터 개수: {total_count}")
    print(f"✅ 중복 없는 원문 개수: {len(content_set)}")

    for post_type, emotion_counter in type_emotion_counter.items():
        total = type_total_counter[post_type]
        print(f"\n--- {post_type} : 총 {total}개 ---")
        for emotion in emotion_order:
            if emotion in emotion_counter:
                print(f"{emotion}: {emotion_counter[emotion]}")
        for emotion, count in emotion_counter.items():
            if emotion not in emotion_order:
                print(f"{emotion}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋 통계 출력")
    parser.add_argument('--data', type=str, required=True, help='입력 JSONL 파일 경로')
    args = parser.parse_args()

    count_posttype_emotion_and_content(args.data)

# python 4.feature_count.py --data my_project/Finetuning/dataset/_dataset/dataset_0615_filtered.jsonl