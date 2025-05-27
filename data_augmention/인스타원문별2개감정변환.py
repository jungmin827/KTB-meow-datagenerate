import json
import random
import os
import argparse
import google.generativeai as genai
from typing import Dict, List, Tuple, Any, Optional
import time
from dotenv import load_dotenv
from collections import Counter

# 환경 변수 로드
load_dotenv()

# Gemini API 키 설정 함수
def setup_gemini_api(api_key: Optional[str] = None) -> None:
    # 우선순위:
    # 1. 함수 인자로 전달된 api_key
    # 2. 환경 변수 GOOGLE_API_KEY
    if api_key is None:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if api_key is None:
            raise ValueError("API 키가 제공되지 않았고 환경 변수 GOOGLE_API_KEY도 설정되지 않았습니다")
    
    genai.configure(api_key=api_key)

# JSONL 파일 쓰기 함수
def write_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# JSON 파일 읽기 함수
def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 단일 텍스트 변환을 위한 프롬프트 생성 함수
def create_transformation_prompt(samples: List[Dict[str, Any]], original_content: str, post_type: str, emotion: str) -> str:
    """단일 텍스트 변환을 위한 프롬프트 생성 함수"""
    # 기본 프롬프트 설정
    base_prompt = f"""
당신은 데이터 증강을 위한 텍스트 변환 시스템입니다.
다음 원본 텍스트를 '{post_type}'(동물 유형)의 '{emotion}'(감정) 스타일로 변환해야 합니다.

원본 텍스트: "{original_content}"

[기본 규칙]
"""
    
    # post_type에 따른 기본 규칙 추가
    if post_type == "cat":
        base_prompt += """
1. 고양이의 말투와 문맥으로 문장을 변환
2. 말투는 반드시 '~냥', '~냐옹', '~다냥', '~옹', '~다옹' 등의 어미로 끝나야 함
3. 'ㅋㅋㅋ'는 '냐하하!', 'ㅎㅎㅎ'는 '먀하하!'로 변환 (각 표현은 한 번만 사용)
4. 고양이 이모티콘(🐈, 🐈‍⬛, 🐾) 중 하나를 전체 글에서 한 번만 사용
5. 불필요한 줄바꿈 없이 자연스럽게 이어지는 문장으로 작성
"""
    elif post_type == "dog":
        base_prompt += """
1. 강아지의 말투와 문맥으로 문장을 변환
2. 말투는 반드시 '~다멍', '~냐왈', '~다컹', '~냐멍', '~다왈', '~다개' 등의 어미로 끝나야 함
3. 강아지 이모티콘(🐕, 🐾, 🦴) 중 하나를 전체 글에서 한 번만 사용
4. 불필요한 줄바꿈 없이 자연스럽게 이어지는 문장으로 작성
"""

    # 감정별 스타일 추가
    if post_type == "cat":
        if emotion == "happy":
            base_prompt += """
[행복한 고양이 스타일]
- 밝고 들뜬 말투 사용
- 하트(❤️), 하트2(💛), 하트3(💙), 웃는 얼굴(ˊᗜˋ), 빛나는(✨) 이모티콘 중 한 개만 맨 뒤에 사용
- 사랑스럽고 신난 느낌을 표현
"""
        elif emotion == "curious":
            base_prompt += """
[호기심 많은 고양이 스타일]
- 궁금한 말투와 장난기 가득한 표현
- 궁금한 표정(=･ｪ･=?), 호기심 가득한(ᓚ₍ ^. .^₎), 신기한(🫨), 궁금한(❓) 이모티콘 중 한 개만 사용
- 킁킁거리며 냄새를 맡거나, 손으로 건드려보는 등 호기심 행동 묘사
"""
        elif emotion == "sad":
            base_prompt += """
[슬픈 고양이 스타일]
- 외로움과 축 처진 말투
- '냐....', '냥....', '옹....' 과 같은 슬픈 문장 표현
- 눈물(😢) 이모티콘 중 한 개만 맨 뒤에 사용
"""
        elif emotion == "grumpy":
            base_prompt += """
[까칠한 고양이 스타일]
- 자신감 넘치는 말투
- 노려봄(=🝦 ༝ 🝦=), 째려봄(𑁢ㅅ𑁢✧), 자신감 넘치는(🐯) 이모티콘 중 한 개만 사용
- 인간을 무시하는 듯한 말투
"""
        elif emotion == "angry":
            base_prompt += """
[화난 고양이 스타일]
- 까칠하고 화난 말투
- 화남(😾, 💢), 불꽃(🔥) 이모티콘 중 한 개만 사용
- '캬아악', '냐아아앙!!!' 같은 의성어 사용
- '냥냥펀치'와 같은 분노 행동 묘사
"""
        elif emotion == "normal":
            base_prompt += """
[일반적인 고양이 스타일]
- 차분하고 평범한 말투
- 고양이스러운 일상적인 표현 사용
- 특별한 감정 없이 일상적인 상황 묘사
"""

    elif post_type == "dog":
        if emotion == "happy":
            base_prompt += """
[행복한 강아지 스타일]
- 밝고 들뜬 말투
- 'ദ്ദി(៸៸›ᴗ‹៸៸ )', '٩(◕ᗜ◕)۶' 중에 하나 선택하여 문장 맨 뒤에 사용
- 신나는 의성어 포함
"""
        elif emotion == "curious":
            base_prompt += """
[호기심 많은 강아지 스타일]
- 궁금해하는 말투
- '(◕ᴥ◕ʋ)?', '⊙﹏⊙', '૮₍◔ᴥ◔₎ა' 중에 하나 선택하여 문장 맨 뒤에 사용
- 호기심 가득한 의성어 포함
"""
        elif emotion == "sad":
            base_prompt += """
[슬픈 강아지 스타일]
- 아무도 자기와 안놀아줘서 슬픈 느낌의 말투
- '૮๑ˊᯅˋ๑ა', '(⊙︿⊙)', '(｡•́︿•̀｡)' 중에 하나 선택하여 문장 맨 뒤에 사용
"""
        elif emotion == "grumpy":
            base_prompt += """
[까칠한 강아지 스타일]
- 고집이 세고 불만이 많은 말투
- '૮ ˙ﻌ˙ ა', 'ᓀ..ᓂ' 중에 하나 선택하여 문장 맨 뒤에 사용
"""
        elif emotion == "angry":
            base_prompt += """
[화난 강아지 스타일]
- 방어적이고 화난 말투
- 'ヾ( ·`⌓´·)ﾉﾞ', '(◣_◢)', '(҂`ﾛ´)' 중에 하나 선택하여 문장 맨 뒤에 사용
- 강아지의 화난 효과음 포함
"""
        elif emotion == "normal":
            base_prompt += """
[일반적인 강아지 스타일]
- 차분하고 평범한 말투
- 강아지스러운 일상적인 표현 사용
- 특별한 감정 없이 일상적인 상황 묘사
"""

    # 예시 데이터 추가
    if samples:
        base_prompt += "\n다음은 기존 데이터의 예시입니다:\n"
        for i, sample in enumerate(samples[:3]):  # 최대 3개 예시만 사용
            base_prompt += f"\n예시 {i+1}:\n원본: {sample['content']}\n변환: {sample['transformed_content']}\n"
    
    # 최종 출력 형식 지정
    base_prompt += f"""
위 예시들을 참고하여 제공된 원본 텍스트를 '{post_type}'(동물)의 '{emotion}'(감정) 스타일로 변환해주세요.
반드시 다음 JSON 형식으로 응답해주세요:

{{
  "content": "{original_content}",
  "emotion": "{emotion}",
  "post_type": "{post_type}",
  "transformed_content": "여기에 변환된 텍스트 작성"
}}

중요: 원본 텍스트는 수정하지 말고, 변환된 텍스트만 작성해주세요.
    - 동물의 감각과 사고방식으로 세상을 바라보고 해석하는 모습을 담아라.
     - 해당 동물의 본능, 습성, 행동 패턴을 자연스럽게 문장에 녹여내라.
     - 동물이 실제로 할 수 있는 행동과 감정 표현에 초점을 맞추어라.
"""
    
    return base_prompt

def parse_single_response(response_text: str, original_content: str, emotion: str, post_type: str) -> Optional[Dict[str, Any]]:
    """단일 응답 파싱 함수"""
    try:
        # 줄바꿈 제거 및 공백 정리
        text = response_text.strip()
        
        # JSON 형식 찾기
        if not text.startswith('{') and not text.endswith('}'):
            # JSON 형식이 아닌 경우 중괄호 사이의 내용 찾기
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
            else:
                return None
        
        # JSON 파싱
        item = json.loads(text)
        
        # 필요한 필드가 모두 있는지 확인
        if all(k in item for k in ['content', 'emotion', 'post_type', 'transformed_content']):
            # 원본 내용, emotion, post_type이 맞는지 확인
            if item['post_type'] == post_type and item['emotion'] == emotion:
                # 원본 내용 교체 (API 응답에서 원본이 변경됐을 수 있음)
                item['content'] = original_content
                
                # 변환된 내용이 비어있지 않은지 확인
                if item['transformed_content'].strip():
                    return item
        
        return None
        
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  응답 파싱 중 오류: {str(e)}")
        return None

def transform_content(model, content: str, post_type: str, emotion: str, samples: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    단일 콘텐츠를 변환하는 함수
    
    :param model: Gemini 모델
    :param content: 원본 콘텐츠
    :param post_type: 포스트 타입 (cat/dog)
    :param emotion: 감정 (happy/sad 등)
    :param samples: 예시 샘플 데이터 (선택 사항)
    :return: 변환된 콘텐츠 또는 None
    """
    max_retries = 3  # 최대 재시도 횟수
    
    # 프롬프트 작성
    prompt = create_transformation_prompt(samples, content, post_type, emotion)
    
    for attempt in range(max_retries):
        try:
            # API 호출
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            if not response or not response.text:
                print(f"  시도 {attempt + 1}: 빈 응답 받음")
                time.sleep(1)
                continue
            
            result_text = response.text
            
            # JSON 객체 파싱
            item = parse_single_response(result_text, content, emotion, post_type)
            
            if item:
                print(f"  성공적으로 변환됨: {post_type} / {emotion}")
                return item
            else:
                print(f"  시도 {attempt + 1}: 유효하지 않은 응답")
                time.sleep(1)
            
        except Exception as e:
            print(f"  API 호출 오류 (시도 {attempt + 1}): {str(e)}")
            time.sleep(1)
    
    return None

def generate_combinations(contents: List[str], target_count: int = 564) -> List[Dict[str, Any]]:
    """
    원본 콘텐츠와 동물/감정 조합을 생성하는 함수
    
    :param contents: 원본 콘텐츠 목록
    :param target_count: 목표 생성 항목 수
    :return: 원본 콘텐츠와 변환 속성을 포함한 리스트
    """
    # 사용 가능한 동물 및 감정 타입
    post_types = ["cat", "dog"]
    emotions = ["happy", "normal", "grumpy", "angry", "curious", "sad"]
    
    # normal과 happy를 조금 더 많이 선택하기 위한 가중치 설정
    emotion_weights = {
        "happy": 0.25,
        "normal": 0.25,
        "grumpy": 0.125,
        "angry": 0.125,
        "curious": 0.125,
        "sad": 0.125
    }
    
    combinations = []
    
    # 각 원본 콘텐츠마다 2개씩의 변환을 생성
    content_index = 0
    
    while len(combinations) < target_count:
        # 원본 콘텐츠 순환 사용
        content = contents[content_index % len(contents)]
        content_index += 1
        
        # 첫 번째 변환 조합 생성
        post_type1 = random.choice(post_types)
        emotion1 = random.choices(emotions, weights=[emotion_weights[e] for e in emotions], k=1)[0]
        
        combinations.append({
            "content": content,
            "post_type": post_type1,
            "emotion": emotion1
        })
        
        # 목표 개수에 도달했는지 확인
        if len(combinations) >= target_count:
            break
        
        # 두 번째 변환 조합 생성 (첫 번째와 다른 조합)
        post_type2 = random.choice(post_types)
        emotion2 = random.choices(emotions, weights=[emotion_weights[e] for e in emotions], k=1)[0]
        
        # 완전히 동일한 조합이 나오지 않도록 확인
        while post_type1 == post_type2 and emotion1 == emotion2:
            post_type2 = random.choice(post_types)
            emotion2 = random.choices(emotions, weights=[emotion_weights[e] for e in emotions], k=1)[0]
        
        combinations.append({
            "content": content,
            "post_type": post_type2,
            "emotion": emotion2
        })
        
        # 목표 개수에 도달했는지 확인
        if len(combinations) >= target_count:
            break
    
    return combinations[:target_count]  # 정확히 목표 개수만큼 반환

def print_distribution(combinations: List[Dict[str, Any]]) -> None:
    """조합의 분포를 출력하는 함수"""
    post_type_count = Counter([item['post_type'] for item in combinations])
    emotion_count = Counter([item['emotion'] for item in combinations])
    
    print("\n==== 생성할 데이터 분포 ====")
    print("\n----- 포스트 타입 분포 -----")
    for post_type, count in post_type_count.items():
        print(f"{post_type}: {count} ({count/len(combinations)*100:.2f}%)")
    
    print("\n----- 감정 분포 -----")
    for emotion, count in emotion_count.items():
        print(f"{emotion}: {count} ({count/len(combinations)*100:.2f}%)")
    
    # 포스트 타입별 감정 분포
    print("\n----- 포스트 타입별 감정 분포 -----")
    for post_type in post_type_count.keys():
        print(f"\n{post_type}:")
        for emotion in emotion_count.keys():
            count = sum(1 for item in combinations if item['post_type'] == post_type and item['emotion'] == emotion)
            print(f"  {emotion}: {count}")

def main():
    parser = argparse.ArgumentParser(description='원문 텍스트를 동물, 감정별로 변환하는 스크립트')
    parser.add_argument('--input_file', type=str, default='데이터/0526sns_content.json', help='입력 JSON 파일 경로')
    parser.add_argument('--output_file', type=str, default='데이터/transformed_content.jsonl', help='출력 JSONL 파일 경로')
    parser.add_argument('--api_key', type=str, help='Google Gemini API 키 (환경 변수 GOOGLE_API_KEY가 설정되어 있지 않은 경우에만 필요)')
    parser.add_argument('--target_count', type=int, default=564, help='목표 생성 항목 수')
    
    args = parser.parse_args()
    
    # API 설정 - 환경 변수나 명령줄 인자를 사용
    try:
        setup_gemini_api(args.api_key)
        print("Google Gemini API가 성공적으로 설정되었습니다.")
    except ValueError as e:
        print(f"오류: {e}")
        print("--api_key 인자를 통해 API 키를 제공하거나 환경 변수 GOOGLE_API_KEY를 설정해주세요.")
        return
    
    # Gemini 모델 설정
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # 데이터 로드
    print(f"데이터 파일 읽는 중: {args.input_file}")
    data = read_json(args.input_file)
    contents = data.get('content', [])
    print(f"로드된 콘텐츠 수: {len(contents)}")
    
    # 원본 콘텐츠와 변환 속성 조합 생성
    combinations = generate_combinations(contents, args.target_count)
    print(f"생성할 조합 수: {len(combinations)}")
    
    # 조합 분포 출력
    print_distribution(combinations)
    
    # 콘텐츠 변환
    transformed_data = []
    batch_size = 10  # 한 번에 처리할 항목 수
    
    # 샘플 데이터 (없음)
    samples = []
    
    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i+batch_size]
        print(f"\n배치 {i//batch_size + 1}/{(len(combinations)-1)//batch_size + 1} 처리 중...")
        
        for j, item in enumerate(batch):
            content = item['content']
            post_type = item['post_type']
            emotion = item['emotion']
            
            print(f"  항목 {i+j+1}/{len(combinations)}: {post_type} / {emotion} 변환 중...")
            
            # 콘텐츠 변환
            transformed_item = transform_content(model, content, post_type, emotion, samples)
            
            if transformed_item:
                transformed_data.append(transformed_item)
                
                # 변환된 항목을 샘플로 추가 (최대 6개까지)
                if len(samples) < 6:
                    samples.append(transformed_item)
            else:
                print(f"  항목 {i+j+1}: 변환 실패, 건너뜀")
        
        # API 속도 제한 방지를 위한 대기
        if i + batch_size < len(combinations):
            print(f"  API 속도 제한 방지를 위해 3초 대기 중...")
            time.sleep(3)
    
    # 결과 저장
    print(f"\n변환된 데이터 저장 중: {args.output_file}")
    write_jsonl(args.output_file, transformed_data)
    print(f"변환 완료: {len(transformed_data)}/{len(combinations)} 항목 변환됨")
    
    # 최종 데이터 분포 출력
    if transformed_data:
        post_type_count = Counter([item['post_type'] for item in transformed_data])
        emotion_count = Counter([item['emotion'] for item in transformed_data])
        
        print("\n==== 최종 데이터 분포 ====")
        print("\n----- 포스트 타입 분포 -----")
        for post_type, count in post_type_count.items():
            print(f"{post_type}: {count} ({count/len(transformed_data)*100:.2f}%)")
        
        print("\n----- 감정 분포 -----")
        for emotion, count in emotion_count.items():
            print(f"{emotion}: {count} ({count/len(transformed_data)*100:.2f}%)")

if __name__ == "__main__":
    main() 