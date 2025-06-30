import json
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일을 읽는 함수"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 빈 줄 무시
                data.append(json.loads(line))
    return data

def analyze_distribution(data: List[Dict[str, Any]]):
    """데이터의 분포를 분석하는 함수"""
    # 포스트 타입과 감정 카운트
    post_type_count = Counter([item['post_type'] for item in data])
    emotion_count = Counter([item['emotion'] for item in data])
    
    # 포스트 타입별 감정 분포
    post_type_emotion_count = {}
    for post_type in post_type_count.keys():
        post_type_emotion_count[post_type] = {}
        for emotion in emotion_count.keys():
            count = sum(1 for item in data if item['post_type'] == post_type and item['emotion'] == emotion)
            post_type_emotion_count[post_type][emotion] = count
    
    return post_type_count, emotion_count, post_type_emotion_count

def print_distribution(post_type_count, emotion_count, post_type_emotion_count, total_count):
    """분포를 출력하는 함수"""
    print("\n==== 데이터 분포 ====")
    print(f"총 데이터 개수: {total_count}")
    
    print("\n----- 포스트 타입 분포 -----")
    for post_type, count in post_type_count.items():
        print(f"{post_type}: {count} ({count/total_count*100:.2f}%)")
    
    print("\n----- 감정 분포 -----")
    for emotion, count in emotion_count.items():
        print(f"{emotion}: {count} ({count/total_count*100:.2f}%)")
    
    print("\n----- 포스트 타입별 감정 분포 -----")
    for post_type, emotions in post_type_emotion_count.items():
        print(f"\n{post_type}:")
        for emotion, count in emotions.items():
            type_total = post_type_count[post_type]
            print(f"  {emotion}: {count} ({count/type_total*100:.2f}%)")

def plot_distribution(post_type_count, emotion_count, post_type_emotion_count, output_prefix=''):
    """분포를 시각화하는 함수"""
    # 포스트 타입 분포 파이 차트
    plt.figure(figsize=(10, 6))
    plt.pie(post_type_count.values(), labels=post_type_count.keys(), autopct='%1.1f%%')
    plt.title('포스트 타입 분포')
    if output_prefix:
        plt.savefig(f'{output_prefix}_post_type_distribution.png')
    else:
        plt.show()
    
    # 감정 분포 바 차트
    plt.figure(figsize=(12, 6))
    plt.bar(emotion_count.keys(), emotion_count.values())
    plt.title('감정 분포')
    plt.ylabel('개수')
    plt.xticks(rotation=45)
    if output_prefix:
        plt.savefig(f'{output_prefix}_emotion_distribution.png')
    else:
        plt.show()
    
    # 포스트 타입별 감정 분포 누적 바 차트
    plt.figure(figsize=(14, 8))
    emotions = list(emotion_count.keys())
    post_types = list(post_type_count.keys())
    
    bottom = [0] * len(post_types)
    for emotion in emotions:
        values = [post_type_emotion_count[pt][emotion] for pt in post_types]
        plt.bar(post_types, values, bottom=bottom, label=emotion)
        bottom = [bottom[i] + values[i] for i in range(len(values))]
    
    plt.title('포스트 타입별 감정 분포')
    plt.ylabel('개수')
    plt.legend()
    if output_prefix:
        plt.savefig(f'{output_prefix}_post_type_emotion_distribution.png')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='JSONL 파일의 포스트 타입과 감정 분포를 분석')
    parser.add_argument('--input_file', type=str, default='데이터/transformed_content.jsonl', help='입력 JSONL 파일 경로')
    parser.add_argument('--output_prefix', type=str, default='', help='출력 이미지 파일 접두사 (지정 시 그래프를 파일로 저장)')
    
    args = parser.parse_args()
    
    # 데이터 로드
    print(f"데이터 파일 읽는 중: {args.input_file}")
    data = read_jsonl(args.input_file)
    print(f"로드된 데이터 수: {len(data)}")
    
    # 데이터 분석
    post_type_count, emotion_count, post_type_emotion_count = analyze_distribution(data)
    
    # 분포 출력
    print_distribution(post_type_count, emotion_count, post_type_emotion_count, len(data))
    
    # 분포 시각화
    plot_distribution(post_type_count, emotion_count, post_type_emotion_count, args.output_prefix)

if __name__ == "__main__":
    main() 