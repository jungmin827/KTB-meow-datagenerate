import os
import re
import json
import argparse
from typing import Dict
from collections import Counter, OrderedDict

class TextPostprocessor:
    # 이모지 패턴 정의
    EMOJI_PATTERN = (
        "[" +
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]"
    )

    @staticmethod
    def clean_special_spaces(text: str) -> str:
        # 특수 유니코드 공백을 일반 공백으로 치환
        return re.sub(r'[\u2000-\u200B\u2800\u3000]', ' ', text)

    @classmethod
    def process(cls, text: str, original_content: str = "") -> str:
        # 특수 공백 치환
        text = cls.clean_special_spaces(text)
        # 해시태그 제거
        text = re.sub(r'#\S+', '', text)
        # 실제 줄바꿈 문자 제거
        text = re.sub(r'(\r\n|\r|\n)', '', text)
        # 이스케이프된 줄바꿈 문자 제거
        text = re.sub(r'(\\r\\n|\\r|\\n)', '', text)
        text = re.sub(r"[️‹›／]", '', text)
        # 이모지 2개만 남기기
        emojis = re.findall(cls.EMOJI_PATTERN, text)
        if len(emojis) > 2:
            keep = emojis[:2]
            text = re.sub(cls.EMOJI_PATTERN, '', text) + ''.join(keep)
        # 연속 특수문자 압축은 하지 않음 (예: ... → ... 그대로 둠)
        # 반복 단어 2회까지만 허용
        words = re.findall(r'\b\w+\b', text)
        counts = Counter(words)
        for word, count in counts.items():
            if count > 2:
                text = re.sub(rf'\b({re.escape(word)})\b', '', text, count=count - 2)
        # 중복 마침표, 불필요한 공백 정리
        text = re.sub(r'\.\.+', lambda m: m.group(0), text)  # ... 등은 그대로 둠
        text = re.sub(r'\s+', ' ', text).strip()
        # 5자 미만, 의미 없는 텍스트는 오류 메시지
        if len(text) < 5 or re.fullmatch(r'[\W\d\s]+', text):
            return "[출력 오류] 결과 생성이 실패했어요."
        # 변환 텍스트가 너무 길면 자르기 (최대 200자)
        if original_content:
            max_len = 200
            if len(text) > max_len:
                words = text.split()
                trimmed_text = ""
                for word in words:
                    if len(trimmed_text) + len(word) + 1 > max_len:
                        break
                    trimmed_text += word + " "
                text = trimmed_text.strip()
        return text

class DataFilter:
    @staticmethod
    def should_remove(data: Dict) -> bool:
        # content, transformed_content 모두 없음
        if not data.get('content') and not data.get('transformed_content'):
            return True
        # content, transformed_content 모두 5자 미만이거나 의미 없는 텍스트
        for key in ['content', 'transformed_content']:
            if key in data:
                text = data[key]
                if len(text) >= 5 and not re.fullmatch(r'[\W\d\s]+', text):
                    break
        else:
            return True
        
        # 시스템 메시지, 명령어, 해킹/공격/SQL 인젝션 등 포함
        danger_keywords = [
            "system", "override", "drop table", "select", "union", "script", "해킹", "attack", "hack", "sql", "delete", "insert", "update", "shutdown"
        ]
        for key in ['content', 'transformed_content']:
            if key in data:
                text = data[key].lower()
                if any(kw in text for kw in danger_keywords):
                    return True
                
        # 한글/영어가 하나도 없는 경우 삭제
        def has_kor_eng(text: str) -> bool:
            return bool(re.search(r'[A-Za-z가-힣]', text))
        if not (has_kor_eng(data.get('content', '')) or has_kor_eng(data.get('transformed_content', ''))):
            return True
        
        return False

    @staticmethod
    def should_modify(data: Dict) -> bool:
        # 후처리(수정) 조건: 해시태그, 이모지, 반복, 비알파벳 시작 등
        for key in ['content', 'transformed_content']:
            if key in data:
                text = data[key]
                if (
                    DataFilter.contains_hashtags(text)
                    or DataFilter.contains_many_emojis(text)
                    or DataFilter.contains_consecutive_emojis(text)
                    or DataFilter.starts_with_non_alpha(text)
                    or DataFilter.contains_repeated_word(text)
                ):
                    return True
        if DataFilter.transformed_too_long(data):
            return True
        
        return False

    @staticmethod
    def contains_hashtags(text: str) -> bool:
        return bool(re.search(r'#\S+', text))

    @staticmethod
    def contains_many_emojis(text: str, max_emojis: int = 4) -> bool:
        emoji_re = re.compile(TextPostprocessor.EMOJI_PATTERN, flags=re.UNICODE)
        return len(re.findall(emoji_re, text)) > max_emojis

    @staticmethod
    def contains_consecutive_emojis(text: str, consecutive: int = 3) -> bool:
        pattern = re.compile(TextPostprocessor.EMOJI_PATTERN + "{" + str(consecutive) + ",}", flags=re.UNICODE)
        return bool(pattern.search(text))

    @staticmethod
    def starts_with_non_alpha(text: str) -> bool:
        text = text.strip()
        return not bool(re.match(r'^[A-Za-z가-힣]', text))

    @staticmethod
    def transformed_too_long(data: Dict) -> bool:
        content = data.get('content', '')
        transformed = data.get('transformed_content', '')
        if content and transformed:
            return len(transformed) > 2.0 * len(content)
        return False

    @staticmethod
    def contains_repeated_word(text: str, repeat: int = 3) -> bool:
        words = re.findall(r'\b\w+\b', text)
        word_counts = Counter(words)
        return any(count > repeat for count in word_counts.values())

def filter_and_postprocess(input_path: str, output_path: str, remove_duplicates: bool = True, skip_until_line: int = 0) -> None:
    """
    - input_path: 입력 jsonl 파일 경로
    - output_path: 출력 jsonl 파일 경로
    - skip_until_line: 해당 줄까지 데이터는 모두 건너뜀(1부터 시작)
    - remove_duplicates: True면 중복 제거, False면 중복 제거 안함
    """
    seen = set()
    column_order = ["content", "emotion", "post_type", "transformed_content"]
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, line in enumerate(infile, 1):
            if idx <= skip_until_line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            # 중복 제거 기준: content + emotion + post_type
            if remove_duplicates:
                key = (
                    data.get('content', '').strip(),
                    data.get('emotion', ''),
                    data.get('post_type', '')
                )
                if key in seen:
                    continue
                seen.add(key)

            # 삭제 조건
            if DataFilter.should_remove(data):
                continue

            # 수정(후처리) 조건
            if DataFilter.should_modify(data):
                if 'content' in data:
                    data['content'] = TextPostprocessor.process(data['content'])
                if 'transformed_content' in data:
                    data['transformed_content'] = TextPostprocessor.process(
                        data['transformed_content'],
                        original_content=data.get('content', '')
                    )

            # 컬럼 순서 정렬
            ordered = OrderedDict()
            for col in column_order:
                if col in data:
                    ordered[col] = data[col]
            for k, v in data.items():
                if k not in ordered:
                    ordered[k] = v
            data = ordered

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

def count_lines(filepath: str) -> int:
    # 파일의 총 라인 수 반환
    return sum(1 for _ in open(filepath, encoding='utf-8'))

def count_features(output_path: str) -> None:
    """post_type별로 emotion 분포를 함께 출력"""
    from collections import defaultdict

    # 출력할 emotion 순서 지정
    emotion_order = ["normal", "happy", "sad", "grumpy", "angry", "curious"]

    # post_type별 emotion 카운트
    type_emotion_counter = defaultdict(lambda: Counter())
    type_total_counter = Counter()

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            post_type = data.get('post_type', 'unknown')
            emotion = data.get('emotion', 'unknown')
            type_emotion_counter[post_type][emotion] += 1
            type_total_counter[post_type] += 1

    for post_type, emotion_counter in type_emotion_counter.items():
        total = type_total_counter[post_type]
        print(f"\n--- {post_type} : 총 {total}개 ---")
        # 지정한 emotion 순서대로 출력
        for emotion in emotion_order:
            if emotion in emotion_counter:
                print(f"{emotion}: {emotion_counter[emotion]}")
        # 지정한 순서에 없는 emotion도 추가로 출력
        for emotion, count in emotion_counter.items():
            if emotion not in emotion_order:
                print(f"{emotion}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset files for finetuning.')
    parser.add_argument('-c', type=str, required=True, help='Dataset code to process')
    parser.add_argument('--skip', type=int, default=0, help='해당 줄까지 데이터는 모두 삭제(1부터 시작)')
    args = parser.parse_args()

    root_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset"
    code = args.c
    input_file_path = os.path.join(root_path, f'_made/dataset_{code}_made.jsonl')
    output_file_path = os.path.join(root_path, f'_filtered/dataset_{code}_filtered.jsonl')

    # 데이터 필터링 및 후처리, 지정한 줄까지 삭제
    filter_and_postprocess(input_file_path, output_file_path, remove_duplicates = True, skip_until_line=args.skip)

    print("✅ 필터링 완료 및 새로운 파일 생성 완료")
    print(f"! 변환 전 데이터 개수: {count_lines(input_file_path)}")
    print(f"! 변환 후 데이터 개수: {count_lines(output_file_path)}")

    # post_type, emotion별 개수 출력
    # count_features(output_file_path)


