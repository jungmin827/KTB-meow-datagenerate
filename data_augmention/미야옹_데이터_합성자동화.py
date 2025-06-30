import json
import random
import os
import argparse
import google.generativeai as genai
from typing import Dict, List, Tuple, Any, Optional, ClassVar
import time
from dotenv import load_dotenv
from collections import Counter
import threading

# 환경 변수 로드
load_dotenv()

class PromptTemplate:
    """프롬프트 템플릿 클래스"""
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """템플릿을 포맷팅하여 최종 프롬프트 반환"""
        return self.template.format(**kwargs)

class AnimalPromptGenerator:
    """동물 스타일 프롬프트 생성 클래스"""
    
    # 동물별 기본 설명
    base_prompt_map: ClassVar[Dict[str, str]] = {
        "cat": """
            [역할] 너는 고양이의 말투와 문맥으로 문장을 재생성하는 변환기다. 
            [규칙]
            1. 문장은 반드시 '~냥', '~냐옹', '~이냥', '~다먀', '~댜옹' 등의 어미로 끝나야 한다.
            2. 'ㅋㅋㅋ'는 '냐하하!'로, 'ㅎㅎㅎ'는 '먀하하!'로 바꾸되, 각 표현은 한 번만 사용하라.
            3. 고양이 기본 이모티콘 정보: 🐈(고양이), 🐾(발자국), 🐈‍⬛(검은 고양이), 🐱(고양이) 이모티콘 중 한개를 골라 전체 글에서 한 번만 사용.
            4. 새로운 문장 생성은 입력 원문 두배 이하로 제한.
            5. 반드시 한국어로만 작성한다.
            """,

        "dog": """
            [역할] 너는 강아지의 말투와 문맥으로 문장을 재생성하는 변환기다. 
            [규칙]
            1. 문장은 반드시 '~멍', '~냐왈', '~다왈', '~다개', '~요멍' 등의 어미로 끝나야 한다.
            2. 반드시 한국어로만 작성한다.
            3. 강아지 기본 이모티콘 정보: 🐩(강아지), 🐕(강아지), 🦴, 🐶(강아지) 이모티콘 중 한개를 골라 전체 글에서 한 번만 사용.
            4. 새로운 문장 생성은 입력 원문 두배 이하로 제한.
            """,
        }

    # 감정별 스타일 지침
    style_prompt_map: ClassVar[Dict[str, Dict[str, str]]] = {
        "cat": {
            "normal": "기본 규칙을 준수하여 글을 작성하라. 평범한 일상의 고양이처럼 느긋하고 여유로운 톤으로 작성.",
            "happy": "밝고 들뜬 말투. \n하트(❤️), 하트2(💛), 하트3(💙),빛나는(✨) 이모티콘 중 한 개만 맨 뒤에 사용.",
            "curious": "궁금해하는 말투. 신기한(🫨), 궁금한(❓) 이모티콘 중 한 개만 문장 맨 뒤에 사용.",
            "sad": "축 처진 말투, 눈물(😢)이모티콘 한 개만 맨 뒤에 사용.",
            "grumpy": "거만한 성격, 고급스러운 말투.",
            "angry": "화났음. 까칠한 말투. \n화남(😾), 화남2(💢), 불꽃(🔥), 이모티콘 중 한 개만 문장 맨 뒤에 사용."
            },
        "dog": {
            "normal": "기본 규칙을 준수하여 글을 작성하라. 평범한 일상에서 즐겁게 지내는 강아지의 느낌으로 작성.",
            "happy": "밝고 들뜬 말투. \n하트(❤️), 하트2(💛), 하트3(💙),빛나는(✨) 이모티콘 중 한 개만 맨 뒤에 사용.",
            "curious": "궁금해하는 말투. 신기한(🫨), 궁금한(❓) 이모티콘 중 한 개만 문장 맨 뒤에 사용.",
            "sad": "풀이 죽은 말투.",
            "grumpy": "불만이 있는 말투",
            "angry": "공격적인 말투. \n화남(😾), 화남2(💢), 불꽃(🔥), 이모티콘 중 한 개만 문장 맨 뒤에 사용."
            }
        }
    
    def __init__(self, content: str, post_type: str, emotion: str):
        self.content = content
        self.post_type = post_type
        self.emotion = emotion
    
    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""  
        template = f"""
{self.base_prompt_map[self.post_type]}

[현재 감정 상태]
{self.emotion}

[감정별 스타일 지침]
{self.style_prompt_map[self.post_type][self.emotion]}

[사용자 입력 원문]
{{content}}

[작성 지침]
- 위 내용을 기반으로, "{self.post_type}"의 말투와 문체로 글을 **일부 재구성**하라.
- 동물의 사고방식으로 세상을 바라보고 해석하는 모습을 담아라.
- 해당 동물의 습성, 행동 패턴을 자연스럽게 문장에 녹여내라.
- 동물이 실제로 할 수 있는 행동과 감정 표현을 넣어라.
- 원문의 단어와 내용은 유지한다.

반드시 다음 JSON 형식으로 응답해주세요:

{{{{
  "content": "{self.content}",
  "emotion": "{self.emotion}",
  "post_type": "{self.post_type}",
  "transformed_content": "여기에 변환된 텍스트 작성"
}}}}
"""

        return PromptTemplate(
            template=template,
            input_variables=["content"]
        )
    
    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content)

class APIKeyPool:
    """API 키 풀링을 관리하는 클래스 (동기식 버전)"""
    
    def __init__(self, api_keys: List[str], max_requests_per_min: int = 15):
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.key_usage = {key: [] for key in api_keys}
        self.key_total_usage = {key: 0 for key in api_keys}  # 총 사용량 추가
        self.max_requests_per_min = max_requests_per_min
        print(f"API 키 풀 초기화 완료: {len(api_keys)}개 키 로드됨")
    
    def _cleanup_old_usage(self, key: str) -> None:
        """1분 이상 지난 사용 기록을 제거합니다."""
        current_time = time.time()
        self.key_usage[key] = [t for t in self.key_usage[key] if current_time - t < 60]
    
    def _is_key_available(self, key: str) -> bool:
        """키의 사용 가능 여부를 확인합니다."""
        self._cleanup_old_usage(key)
        return len(self.key_usage[key]) < self.max_requests_per_min
    
    def get_available_key(self) -> Optional[str]:
        """사용 가능한 API 키를 반환하고 다음 키로 순환합니다."""
        with self.lock:
            if not self.api_keys:
                return None
            
            # 가장 적게 사용된 키부터 확인
            available_keys = []
            for i, key in enumerate(self.api_keys):
                self._cleanup_old_usage(key)
                usage = len(self.key_usage[key])
                if usage < self.max_requests_per_min:
                    available_keys.append((i, key, usage))
            
            if not available_keys:
                # 모든 키가 제한에 걸린 경우
                print("  🚫 모든 API 키가 분당 제한에 도달했습니다. 대기가 필요합니다.")
                return None
            
            # 사용량이 가장 적은 키 선택 (라운드로빈 + 로드밸런싱)
            available_keys.sort(key=lambda x: x[2])  # 사용량 기준 정렬
            key_index, selected_key, current_usage = available_keys[0]
            
            # 키 사용 기록 추가
            self.key_usage[selected_key].append(time.time())
            self.key_total_usage[selected_key] += 1  # 총 사용량 증가
            new_usage = current_usage + 1
            
            print(f"  ✅ 키 #{key_index + 1} 사용 (현재: {new_usage}/{self.max_requests_per_min}/분, 총: {self.key_total_usage[selected_key]}회)")
            
            # 다음 순환을 위해 인덱스 업데이트 (선택적)
            self.current_index = (key_index + 1) % len(self.api_keys)
            
            return selected_key
    
    def get_status(self) -> str:
        """현재 키 풀 상태를 반환합니다."""
        status_lines = []
        for i, key in enumerate(self.api_keys):
            self._cleanup_old_usage(key)
            current_usage = len(self.key_usage[key])
            total_usage = self.key_total_usage[key]
            status = f"키 #{i+1}: {current_usage}/{self.max_requests_per_min}/분 (총:{total_usage})"
            status_lines.append(status)
        return " | ".join(status_lines)

def load_api_keys_from_env() -> List[str]:
    """환경 변수에서 API 키들을 로드합니다."""
    api_keys = []
    
    # GOOGLE_API_KEY (단일 키)
    single_key = os.environ.get('GOOGLE_API_KEY')
    if single_key:
        api_keys.append(single_key)
    
    # GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3 (복수 키)
    for i in range(1, 10):  # 최대 9개까지 확인
        key = os.environ.get(f'GOOGLE_API_KEY_{i}')
        if key:
            api_keys.append(key)
    
    # GOOGLE_API_KEYS (콤마로 구분된 여러 키)
    keys_str = os.environ.get('GOOGLE_API_KEYS')
    if keys_str:
        additional_keys = [key.strip() for key in keys_str.split(',') if key.strip()]
        api_keys.extend(additional_keys)
    
    # 중복 제거
    api_keys = list(set(api_keys))
    
    return api_keys

def initialize_key_pool() -> APIKeyPool:
    """API 키 풀을 초기화합니다."""
    api_keys = load_api_keys_from_env()
    
    if not api_keys:
        raise ValueError("API 키가 환경 변수에서 발견되지 않았습니다. GOOGLE_API_KEY, GOOGLE_API_KEY_1~9, 또는 GOOGLE_API_KEYS를 설정해주세요.")
    
    print(f"총 {len(api_keys)}개의 API 키를 발견했습니다.")
    return APIKeyPool(api_keys=api_keys, max_requests_per_min=15)

# 전역 키 풀 인스턴스
key_pool: Optional[APIKeyPool] = None

# Gemini API 키 설정 함수
def setup_gemini_api(api_key: Optional[str] = None) -> None:
    global key_pool
    
    if api_key is None:
        # 키 풀 초기화
        key_pool = initialize_key_pool()
        print(f"✅ API 키 풀이 성공적으로 설정되었습니다. ({key_pool.get_status()})")
    else:
        # 단일 키 사용
        genai.configure(api_key=api_key)
        print("✅ 단일 API 키가 성공적으로 설정되었습니다.")

# JSONL 파일 읽기 함수
def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일을 읽어서 리스트로 반환"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# JSONL 파일 쓰기 함수
def write_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_transformation_prompt(original_content: str, post_type: str, emotion: str) -> str:
    """새로운 클래스 기반 프롬프트 생성 함수"""
    generator = AnimalPromptGenerator(original_content, post_type, emotion)
    return generator.get_formatted_prompt()

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

def transform_content(model, content: str, post_type: str, emotion: str) -> Optional[Dict[str, Any]]:
    """
    단일 콘텐츠를 변환하는 함수 (키 풀 지원)
    
    :param model: Gemini 모델 클래스 (genai.GenerativeModel)
    :param content: 원본 콘텐츠
    :param post_type: 포스트 타입 (cat/dog)
    :param emotion: 감정 (happy/sad 등)
    :return: 변환된 콘텐츠 또는 None
    """
    global key_pool
    max_retries = 3  # 최대 재시도 횟수
    
    # 프롬프트 작성
    prompt = create_transformation_prompt(content, post_type, emotion)
    
    for attempt in range(max_retries):
        try:
            # API 키 선택
            if key_pool:
                api_key = key_pool.get_available_key()
                if api_key is None:
                    print(f"  ⏳ 모든 키 제한 도달. 60초 대기...")
                    time.sleep(60)  # 1분 대기 후 재시도
                    continue
                
                # 선택된 키로 모델 설정
                genai.configure(api_key=api_key)
                current_model = genai.GenerativeModel('gemini-2.0-flash')
            else:
                current_model = model
            
            # API 호출
            response = current_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 3,
                    "max_output_tokens": 400
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
                print(f"  성공: {post_type}/{emotion}")
                return item
            else:
                print(f"  시도 {attempt + 1}: 유효하지 않은 응답")
                time.sleep(1)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"  API 제한 오류 (시도 {attempt + 1}): 키 변경 후 재시도")
                if key_pool:
                    print(f"  현재 키 상태: {key_pool.get_status()}")
                time.sleep(5)  # 5초 대기 후 다른 키로 재시도
            else:
                print(f"  API 호출 오류 (시도 {attempt + 1}): {error_msg}")
                time.sleep(2)
    
    return None

def generate_all_combinations(contents: List[str]) -> List[Dict[str, Any]]:
    """
    모든 원본 콘텐츠에 대해 모든 동물/감정 조합을 생성하는 함수
    
    :param contents: 원본 콘텐츠 목록
    :return: 모든 조합을 포함한 리스트
    """
    # 사용 가능한 동물 및 감정 타입
    post_types = ["cat", "dog"]
    emotions = ["happy", "normal", "grumpy", "angry", "curious", "sad"]
    
    combinations = []
    
    # 각 원본 콘텐츠마다 모든 조합 생성
    for content in contents:
        for post_type in post_types:
            for emotion in emotions:
                combinations.append({
                    "content": content,
                    "post_type": post_type,
                    "emotion": emotion
                })
    
    return combinations

def print_distribution(combinations: List[Dict[str, Any]]) -> None:
    """조합의 분포를 출력하는 함수"""
    post_type_count = Counter([item['post_type'] for item in combinations])
    emotion_count = Counter([item['emotion'] for item in combinations])
    
    print("\n==== 생성할 데이터 분포 ====")
    print(f"총 조합 수: {len(combinations)}")
    
    print("\n----- 포스트 타입 분포 -----")
    for post_type, count in post_type_count.items():
        print(f"{post_type}: {count} ({count/len(combinations)*100:.2f}%)")
    
    print("\n----- 감정 분포 -----")
    for emotion, count in emotion_count.items():
        print(f"{emotion}: {count} ({count/len(combinations)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='미야옹 원문 텍스트를 동물, 감정별로 대량 변환하는 스크립트')
    parser.add_argument('--input_file', type=str, default='데이터/0617합성한원문들.jsonl', help='입력 JSONL 파일 경로')
    parser.add_argument('--output_file', type=str, default='데이터/0618미야옹_합성_데이터.jsonl', help='출력 JSONL 파일 경로')
    parser.add_argument('--api_key', type=str, help='Google Gemini API 키 (환경 변수 GOOGLE_API_KEY가 설정되어 있지 않은 경우에만 필요)')
    parser.add_argument('--batch_size', type=int, default=10, help='배치 크기 (한 번에 처리할 항목 수)')
    parser.add_argument('--sleep_time', type=int, default=4, help='배치 간 대기 시간 (초)')
    
    args = parser.parse_args()
    
    # API 설정
    try:
        setup_gemini_api(args.api_key)
        if key_pool:
            print(f"🔑 사용 가능한 API 키: {len(key_pool.api_keys)}개")
            print(f"📊 분당 제한: 키당 {key_pool.max_requests_per_min}회")
            print(f"⚡ 예상 처리량: 분당 최대 {len(key_pool.api_keys) * key_pool.max_requests_per_min}회")
        else:
            print("Google Gemini API가 성공적으로 설정되었습니다.")
    except ValueError as e:
        print(f"오류: {e}")
        print("--api_key 인자를 통해 API 키를 제공하거나 환경 변수를 설정해주세요.")
        return
    
    # Gemini 모델 설정
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # 데이터 로드
    print(f"데이터 파일 읽는 중: {args.input_file}")
    data = read_jsonl(args.input_file)
    contents = [item['content'] for item in data]
    print(f"로드된 콘텐츠 수: {len(contents)}")
    
    # 모든 조합 생성
    combinations = generate_all_combinations(contents)
    print(f"생성할 조합 수: {len(combinations)}")
    
    # 조합 분포 출력
    print_distribution(combinations)
    
    # 콘텐츠 변환
    transformed_data = []
    failed_count = 0
    
    print(f"\n==== 대량 데이터 변환 시작 ====")
    
    for i in range(0, len(combinations), args.batch_size):
        batch = combinations[i:i+args.batch_size]
        batch_num = i//args.batch_size + 1
        total_batches = (len(combinations)-1)//args.batch_size + 1
        
        print(f"\n배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 항목)")
        
        for j, item in enumerate(batch):
            content = item['content']
            post_type = item['post_type']
            emotion = item['emotion']
            
            current_item = i + j + 1
            print(f"  [{current_item}/{len(combinations)}] {post_type}/{emotion} 변환 중...")
            
            # 콘텐츠 변환
            transformed_item = transform_content(model, content, post_type, emotion)
            
            if transformed_item:
                transformed_data.append(transformed_item)
            else:
                failed_count += 1
                print(f"  [{current_item}] 변환 실패")
        
        # 배치별 진행상황 출력
        success_rate = ((len(transformed_data) / current_item) * 100) if current_item > 0 else 0
        print(f"  배치 {batch_num} 완료 - 성공: {len(transformed_data)}, 실패: {failed_count}, 성공률: {success_rate:.1f}%")
        
        # 키 풀 상태 출력
        if key_pool:
            print(f"  🔑 키 상태: {key_pool.get_status()}")
        
        # API 속도 제한 방지를 위한 대기
        if i + args.batch_size < len(combinations):
            print(f"  API 속도 제한 방지를 위해 {args.sleep_time}초 대기 중...")
            time.sleep(args.sleep_time)
        
        # 중간 저장 (100개 배치마다)
        if batch_num % 10 == 0:
            temp_output_file = f"{args.output_file}.temp_{batch_num}"
            write_jsonl(temp_output_file, transformed_data)
            print(f"  중간 저장 완료: {temp_output_file}")
    
    # 최종 결과 저장
    print(f"\n변환된 데이터 저장 중: {args.output_file}")
    write_jsonl(args.output_file, transformed_data)
    
    # 최종 통계 출력
    print(f"\n==== 변환 완료 ====")
    print(f"총 처리 항목: {len(combinations)}")
    print(f"성공적으로 변환됨: {len(transformed_data)}")
    print(f"변환 실패: {failed_count}")
    print(f"성공률: {(len(transformed_data)/len(combinations)*100):.1f}%")
    
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