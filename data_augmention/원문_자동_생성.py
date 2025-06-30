import json
import random
import os
import argparse
import google.generativeai as genai
from typing import Dict, List, Tuple, Any, Optional
import time
from dotenv import load_dotenv
import threading
import re

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

class ContentGenerator:
    """원문 생성 클래스"""
    
    # 주제 목록
    TOPICS = [
        "출근", "퇴근", "점심", "배고픔", "고양이", "모의면접", "맛집","회식", "늦잠", "사진",
        "동물", "졸림", "행복", "귀여움", "미스코리냥", "커피", "운동", "날씨", "귀엽다", "심심하다", "칼퇴", "면접", "지하철"
    ]
    
    # 감정 목록 (한국어 -> 영어 매핑)
    EMOTION_MAP = {
        "일반": "normal",
        "행복": "happy",
        "슬픔": "sad",
        "화남": "angry",
        "삐짐": "grumpy",
        "신남": "curious"
    }
    
    # 기본 프롬프트 템플릿
    BASE_PROMPT = """
    너는 SNS 서비스의 사용자야.  
    이 서비스의 사용자들은 일상적인 이야기, 감정 표현, 귀여운 말투, 재치 있는 문장을 자주 사용해.  
    말투는 존댓말이야. 예시처럼 실제 원문들을 적어.
    짧고 캐주얼한 톤으로, 1~3문장 내외의 문장을 생성해줘.  

    아래는 실제 사용자들의 예시야.

    ### 예시 1
    주제: 배고픔  
    감정: 화남  
    원문: "원래 아침을 먹는데 아침을 못먹고 와서 정말 배고파 죽겠다"

    ### 예시 2  
    주제: 졸림  
    감정: 일반  
    원문: "나른나른 잠와"

    ### 예시 3  
    주제: 날씨  
    감정: 삐짐 
    원문: "밖에 비가오는데 친구가 우산을 안씌워줘서 서운했어요"

    ### 예시 4  
    주제: 운동
    감정: 행복
    원문: "운동 갔다가 거울 보니 살짝 라인이 잡힌 것 같아서 뿌듯했어."

    ### 예시 5  
    주제: 미스코리냥 
    감정: 신남
    원문: "지난주 미스코리냥들 축하해~!! 내가 이번주 미스코리냥을 노려보겠어!! 오늘 날씨 너무 좋던데 외출하기 전에 셀카 한장 남긴다ㅎㅎ 다들 뽀얀 피부 유지하려면 모자 필수인거 알지~??"

    ### 예시 6
    주제: 회식
    감정: 슬픔
    원문: "회식이 있었다. 기가 너무 빨렸다. 얼른 집가고싶다."

    ---

    이제 아래 조건을 참고해서 새로운 원문을 생성해줘.  
    조건:
    - 주제: {topic1}, {topic2} (두 주제를 자연스럽게 연결하거나 하나만 선택해서 사용해도 됨)
    - 감정: {emotion}
    - 문장 길이: 1~3문장 (최대 80자 이내)

    반드시 다음 JSON 형식으로 응답해주세요:
    {{
        "content": "여기에 생성된 텍스트 작성",
        "topic": "{topic1}, {topic2}",
        "emotion": "{emotion}"
    }}
    """
    
    def __init__(self):
        self.template = PromptTemplate(
            template=self.BASE_PROMPT,
            input_variables=["topic1", "topic2", "emotion"]
        )
    
    def get_random_topics(self) -> Tuple[str, str]:
        """랜덤 주제 두 개 선택"""
        return tuple(random.sample(self.TOPICS, 2))
    
    def get_random_emotion(self) -> str:
        """랜덤 감정 선택"""
        return random.choice(list(self.EMOTION_MAP.keys()))
    
    def get_english_emotion(self, korean_emotion: str) -> str:
        """한국어 감정을 영어로 변환"""
        return self.EMOTION_MAP.get(korean_emotion, "normal")
    
    def create_prompt(self, topic1: str, topic2: str, emotion: str) -> str:
        """프롬프트 생성"""
        return self.template.format(topic1=topic1, topic2=topic2, emotion=emotion)

class APIKeyPool:
    """API 키 풀링을 관리하는 클래스"""
    
    def __init__(self, api_keys: List[str], max_requests_per_min: int = 15):
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.key_usage = {key: [] for key in api_keys}
        self.key_total_usage = {key: 0 for key in api_keys}
        self.max_requests_per_min = max_requests_per_min
        print(f"API 키 풀 초기화 완료: {len(api_keys)}개 키 로드됨")
    
    def _cleanup_old_usage(self, key: str) -> None:
        """1분 이상 지난 사용 기록을 제거"""
        current_time = time.time()
        self.key_usage[key] = [t for t in self.key_usage[key] if current_time - t < 60]
    
    def get_available_key(self) -> Optional[str]:
        """사용 가능한 API 키를 반환"""
        with self.lock:
            if not self.api_keys:
                return None
            
            available_keys = []
            for i, key in enumerate(self.api_keys):
                self._cleanup_old_usage(key)
                usage = len(self.key_usage[key])
                if usage < self.max_requests_per_min:
                    available_keys.append((i, key, usage))
            
            if not available_keys:
                print("  🚫 모든 API 키가 분당 제한에 도달했습니다. 대기가 필요합니다.")
                return None
            
            available_keys.sort(key=lambda x: x[2])
            key_index, selected_key, current_usage = available_keys[0]
            
            self.key_usage[selected_key].append(time.time())
            self.key_total_usage[selected_key] += 1
            new_usage = current_usage + 1
            
            print(f"  ✅ 키 #{key_index + 1} 사용 (현재: {new_usage}/{self.max_requests_per_min}/분, 총: {self.key_total_usage[selected_key]}회)")
            
            self.current_index = (key_index + 1) % len(self.api_keys)
            
            return selected_key
    
    def get_status(self) -> str:
        """현재 키 풀 상태를 반환"""
        status_lines = []
        for i, key in enumerate(self.api_keys):
            self._cleanup_old_usage(key)
            current_usage = len(self.key_usage[key])
            total_usage = self.key_total_usage[key]
            status = f"키 #{i+1}: {current_usage}/{self.max_requests_per_min}/분 (총:{total_usage})"
            status_lines.append(status)
        return " | ".join(status_lines)

def load_api_keys_from_env() -> List[str]:
    """환경 변수에서 API 키들을 로드"""
    api_keys = []
    
    single_key = os.environ.get('GOOGLE_API_KEY')
    if single_key:
        api_keys.append(single_key)
    
    for i in range(1, 10):
        key = os.environ.get(f'GOOGLE_API_KEY_{i}')
        if key:
            api_keys.append(key)
    
    keys_str = os.environ.get('GOOGLE_API_KEYS')
    if keys_str:
        additional_keys = [key.strip() for key in keys_str.split(',') if key.strip()]
        api_keys.extend(additional_keys)
    
    api_keys = list(set(api_keys))
    
    return api_keys

def initialize_key_pool() -> APIKeyPool:
    """API 키 풀을 초기화"""
    api_keys = load_api_keys_from_env()
    
    if not api_keys:
        raise ValueError("API 키가 환경 변수에서 발견되지 않았습니다. GOOGLE_API_KEY, GOOGLE_API_KEY_1~9, 또는 GOOGLE_API_KEYS를 설정해주세요.")
    
    print(f"총 {len(api_keys)}개의 API 키를 발견했습니다.")
    return APIKeyPool(api_keys=api_keys, max_requests_per_min=15)

def setup_gemini_api(api_key: Optional[str] = None) -> None:
    """Gemini API 설정"""
    global key_pool
    
    if api_key is None:
        key_pool = initialize_key_pool()
        print(f"✅ API 키 풀이 성공적으로 설정되었습니다. ({key_pool.get_status()})")
    else:
        genai.configure(api_key=api_key)
        print("✅ 단일 API 키가 성공적으로 설정되었습니다.")

def write_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    """JSONL 파일 쓰기"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # content와 emotion을 같은 레벨에 저장
            output_item = {
                "content": item['content'],
                "emotion": item['emotion']
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

def generate_content(model, generator: ContentGenerator) -> Optional[Dict[str, Any]]:
    """단일 콘텐츠 생성"""
    global key_pool
    max_retries = 3
    
    topic1, topic2 = generator.get_random_topics()
    emotion = generator.get_random_emotion()
    prompt = generator.create_prompt(topic1, topic2, emotion)
    
    for attempt in range(max_retries):
        try:
            if key_pool:
                api_key = key_pool.get_available_key()
                if api_key is None:
                    print(f"  ⏳ 모든 키 제한 도달. 60초 대기...")
                    time.sleep(60)
                    continue
                
                genai.configure(api_key=api_key)
                current_model = genai.GenerativeModel('gemini-2.0-flash')
            else:
                current_model = model
            
            response = current_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 2,
                    "max_output_tokens": 200
                }
            )
            
            if not response or not response.text:
                print(f"  시도 {attempt + 1}: 빈 응답 받음")
                time.sleep(1)
                continue
            
            try:
                # 응답 텍스트에서 JSON 부분만 추출
                text = response.text.strip()
                
                # JSON 형식 찾기
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if not json_match:
                    print(f"  시도 {attempt + 1}: JSON 형식을 찾을 수 없음")
                    print(f"  응답: {text[:100]}...")
                    time.sleep(1)
                    continue
                
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                if all(k in result for k in ['content', 'topic', 'emotion']):
                    # 원문과 감정이 매칭되도록 수정
                    result = {
                        "content": result['content'],
                        "emotion": generator.get_english_emotion(result['emotion'])
                    }
                    print(f"  성공: {topic1},{topic2}/{emotion} -> {result['emotion']}")
                    return result
                else:
                    print(f"  시도 {attempt + 1}: 필수 키 누락")
                    print(f"  응답: {json_str}")
                    time.sleep(1)
                    
            except json.JSONDecodeError as e:
                print(f"  시도 {attempt + 1}: JSON 파싱 실패 - {str(e)}")
                print(f"  응답: {text[:100]}...")
                time.sleep(1)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"  API 제한 오류 (시도 {attempt + 1}): 키 변경 후 재시도")
                if key_pool:
                    print(f"  현재 키 상태: {key_pool.get_status()}")
                time.sleep(5)
            else:
                print(f"  API 호출 오류 (시도 {attempt + 1}): {error_msg}")
                time.sleep(2)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='반려동물 SNS 스타일의 원문을 자동으로 생성하는 스크립트')
    parser.add_argument('--output_file', type=str, default='데이터/원문_자동_생성.jsonl', help='출력 JSONL 파일 경로')
    parser.add_argument('--api_key', type=str, help='Google Gemini API 키')
    parser.add_argument('--num_samples', type=int, default=20, help='생성할 문장 수')
    parser.add_argument('--batch_size', type=int, default=5, help='배치 크기')
    parser.add_argument('--sleep_time', type=int, default=3, help='배치 간 대기 시간 (초)')
    
    args = parser.parse_args()
    
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
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    generator = ContentGenerator()
    
    generated_data = []
    failed_count = 0
    
    print(f"\n==== 원문 자동 생성 시작 (목표: {args.num_samples}개) ====")
    
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - i)
        batch_num = i//args.batch_size + 1
        total_batches = (args.num_samples-1)//args.batch_size + 1
        
        print(f"\n배치 {batch_num}/{total_batches} 처리 중... ({batch_size}개 항목)")
        
        for j in range(batch_size):
            current_item = i + j + 1
            print(f"  [{current_item}/{args.num_samples}] 생성 중...")
            
            result = generate_content(model, generator)
            
            if result:
                generated_data.append(result)
            else:
                failed_count += 1
                print(f"  [{current_item}] 생성 실패")
        
        success_rate = ((len(generated_data) / current_item) * 100) if current_item > 0 else 0
        print(f"  배치 {batch_num} 완료 - 성공: {len(generated_data)}, 실패: {failed_count}, 성공률: {success_rate:.1f}%")
        
        if key_pool:
            print(f"  🔑 키 상태: {key_pool.get_status()}")
        
        if i + args.batch_size < args.num_samples:
            print(f"  API 속도 제한 방지를 위해 {args.sleep_time}초 대기 중...")
            time.sleep(args.sleep_time)
        
        if batch_num % 5 == 0:
            temp_output_file = f"{args.output_file}.temp_{batch_num}"
            write_jsonl(temp_output_file, generated_data)
            print(f"  중간 저장 완료: {temp_output_file}")
    
    print(f"\n생성된 데이터 저장 중: {args.output_file}")
    write_jsonl(args.output_file, generated_data)
    
    print(f"\n==== 생성 완료 ====")
    print(f"총 생성 시도: {args.num_samples}")
    print(f"성공적으로 생성됨: {len(generated_data)}")
    print(f"생성 실패: {failed_count}")
    print(f"성공률: {(len(generated_data)/args.num_samples*100):.1f}%")
    
    if generated_data:
        topic_count = {}
        emotion_count = {}
        
        for item in generated_data:
            topic = item['topic']
            emotion = item['emotion']
            topic_count[topic] = topic_count.get(topic, 0) + 1
            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
        
        print("\n==== 생성된 데이터 분포 ====")
        print("\n----- 주제 분포 -----")
        for topic, count in topic_count.items():
            print(f"{topic}: {count} ({count/len(generated_data)*100:.2f}%)")
        
        print("\n----- 감정 분포 -----")
        for emotion, count in emotion_count.items():
            print(f"{emotion}: {count} ({count/len(generated_data)*100:.2f}%)")

if __name__ == "__main__":
    main() 