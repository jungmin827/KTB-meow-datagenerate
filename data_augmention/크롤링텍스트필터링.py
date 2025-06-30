import json
import google.generativeai as genai
import os
import time
from tqdm import tqdm
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 명령행 인자 파싱을 위한 설정
parser = argparse.ArgumentParser(description='고양이-강아지 문장 변환에 적합한 텍스트를 배치 처리합니다.')
parser.add_argument('--input_file', type=str, default='0523captions_only.json', help='입력 JSON 파일 경로')
parser.add_argument('--output_file', type=str, default='filtered_cat_dog.json', help='출력 JSON 파일 경로')
parser.add_argument('--api_key', type=str, required=True, help='Google Gemini API 키')
parser.add_argument('--num_samples', type=int, default=300, help='필터링할 샘플 수')
parser.add_argument('--batch_size', type=int, default=10, help='한 번에 처리할 배치 크기')
parser.add_argument('--max_workers', type=int, default=5, help='병렬 처리에 사용할 최대 워커 수')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='중간 결과를 저장할 간격')
parser.add_argument('--max_length', type=int, default=200, help='최대 문장 길이 (기본값: 200자)')
args = parser.parse_args()

# API 키 설정
genai.configure(api_key=args.api_key)

# 모델 설정
model = genai.GenerativeModel('gemini-1.5-pro')

# 체크포인트 파일 경로
CHECKPOINT_FILE = f"checkpoint_{args.output_file}"

def load_json_data(file_path):
    """JSON 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_data(data, file_path):
    """데이터를 JSON 파일로 저장합니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def prefilter_texts(data):
    """
    기본적인 규칙을 사용하여 텍스트를 사전 필터링합니다.
    이는 API 호출을 줄이는 데 도움이 됩니다.
    """
    prefiltered = []
    
    for text in data:
        # 문자열이 아니면 건너뜁니다
        if not isinstance(text, str):
            continue
        
        # 너무 짧은 텍스트 건너뜁니다
        if len(text.strip()) < 20:
            continue
        
        # 텍스트가 최대 길이를 초과하면 건너뜁니다
        if len(text) > args.max_length:
            continue
        
        # 해시태그가 너무 많거나 텍스트보다 많으면 건너뜁니다
        if text.count('#') > 10 or text.count('#') > text.count(' ') / 2:
            continue
        
        # 고양이 관련 키워드가 포함되어 있는지 확인합니다
        cat_keywords = ['고양이', '냥이', '강아지', '냥', '야옹', '집사', '멍', '묘']
        if not any(keyword in text.lower() for keyword in cat_keywords):
            continue
        
        # 특정 키워드가 포함된 텍스트는 제외합니다
        exclude_keywords = ['임보', '임시보호', '평생가족', '협찬']
        if any(keyword in text for keyword in exclude_keywords):
            continue
            
        # 영어 비율이 높은 텍스트는 제외합니다 (약 30% 이상)
        english_chars = sum(1 for c in text if ord('a') <= ord(c.lower()) <= ord('z'))
        if english_chars > len(text) * 0.3:
            continue
        
        prefiltered.append(text)
    
    logger.info(f"사전 필터링: {len(data)}개 중 {len(prefiltered)}개의 텍스트가 남았습니다.")
    return prefiltered

def is_suitable_for_transformation(text):
    """
    해당 텍스트가 고양이-강아지 문장 변환에 적합한지 평가합니다.
    """
    prompt = f"""
    아래 텍스트가 고양이-강아지 문장 변환에 적합한지 평가해주세요.
    
    텍스트: "{text}"
    
    평가 기준:
    1. 고양이의 행동, 특성, 습관 등을 명확하게 묘사하고 있는가?
    3. 영어가 없어야 함
    4. 임보, 임시보호, 평생가족, 협찬이라는 단어가 없어야 함
    5. 충분한 문맥과 내용이 있는가? (단순 태그나 단어 나열이 아닌가?)
    6. 해시태그가 없거나 매우 적어야 함
    
    위 기준을 바탕으로, 이 텍스트가 고양이-강아지 변환에 적합하면 True, 그렇지 않으면 False로만 답해주세요.
    """
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            result = response.text.strip().lower()
            
            # 'true' 또는 'false'만 추출
            if 'true' in result:
                return True
            else:
                return False
        except Exception as e:
            retry_count += 1
            logger.warning(f"API 호출 오류 (시도 {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                # 지수 백오프
                time.sleep(2 ** retry_count)
            else:
                logger.error(f"최대 재시도 횟수 초과: {text[:50]}...")
                return False

def batch_filter_worker(texts_batch):
    """
    텍스트 배치를 필터링하는 워커 함수입니다.
    """
    suitable_texts = []
    
    for text in texts_batch:
        if is_suitable_for_transformation(text):
            suitable_texts.append(text)
    
    return suitable_texts

def load_checkpoint():
    """체크포인트 파일을 로드합니다."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"체크포인트 로드 중 오류: {e}")
    return {"filtered_texts": [], "processed_count": 0}

def save_checkpoint(filtered_texts, processed_count):
    """체크포인트 파일을 저장합니다."""
    checkpoint_data = {
        "filtered_texts": filtered_texts,
        "processed_count": processed_count
    }
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        logger.info(f"체크포인트 저장: {len(filtered_texts)}개 텍스트, {processed_count}개 처리됨")
    except Exception as e:
        logger.error(f"체크포인트 저장 중 오류: {e}")

def batch_process_texts():
    """
    텍스트를 배치 처리합니다.
    """
    logger.info("고양이-강아지 문장 변환에 적합한 텍스트 필터링을 시작합니다.")
    logger.info(f"최대 문장 길이: {args.max_length}자")
    
    # 데이터 로드
    data = load_json_data(args.input_file)
    logger.info(f"{len(data)}개의 텍스트를 로드했습니다.")
    
    # 체크포인트 확인
    checkpoint = load_checkpoint()
    filtered_texts = checkpoint["filtered_texts"]
    processed_count = checkpoint["processed_count"]
    
    if filtered_texts:
        logger.info(f"체크포인트에서 {len(filtered_texts)}개의 필터링된 텍스트와 {processed_count}개의 처리된 텍스트를 로드했습니다.")
    
    # 사전 필터링
    if processed_count == 0:
        data = prefilter_texts(data)
    else:
        # 이미 처리된 항목 건너뛰기
        data = data[processed_count:]
        logger.info(f"이미 처리된 {processed_count}개의 텍스트를 건너뜁니다.")
    
    # 필요한 추가 샘플 수 계산
    remaining_samples = args.num_samples - len(filtered_texts)
    
    if remaining_samples <= 0:
        logger.info(f"이미 충분한 텍스트({len(filtered_texts)}개)가 필터링되었습니다.")
        return filtered_texts
    
    logger.info(f"추가로 {remaining_samples}개의 텍스트를 필터링합니다...")
    
    # 프로그레스 바 설정
    with tqdm(total=remaining_samples) as pbar:
        pbar.update(len(filtered_texts))
        
        # 배치로 데이터 처리
        batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            if len(filtered_texts) >= args.num_samples:
                break
            
            # 배치를 다시 작은 그룹으로 나누어 병렬 처리
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 각 워커당 1-2개의 텍스트 할당
                worker_batch_size = max(1, len(batch) // args.max_workers)
                worker_batches = [batch[i:i + worker_batch_size] for i in range(0, len(batch), worker_batch_size)]
                
                results = list(executor.map(batch_filter_worker, worker_batches))
            
            # 결과 합치기
            for result in results:
                filtered_texts.extend(result)
                pbar.update(len(result))
                
                if len(filtered_texts) >= args.num_samples:
                    break
            
            # 처리된 텍스트 수 업데이트
            processed_count += len(batch)
            
            # 정기적으로 체크포인트 저장
            if (batch_idx + 1) % args.checkpoint_interval == 0:
                save_checkpoint(filtered_texts[:args.num_samples], processed_count)
    
    # 최종 결과 저장
    final_filtered_texts = filtered_texts[:args.num_samples]
    
    logger.info(f"필터링 완료: {len(final_filtered_texts)}개의 적합한 텍스트를 찾았습니다.")
    
    return final_filtered_texts

def main():
    # 배치 처리
    filtered_texts = batch_process_texts()
    
    # 최종 결과 저장
    result = {
        "filtered_texts": filtered_texts,
        "total_original": len(load_json_data(args.input_file)),
        "total_filtered": len(filtered_texts)
    }
    
    save_json_data(result, args.output_file)
    logger.info(f"필터링된 텍스트를 {args.output_file}에 저장했습니다.")
    
    # 체크포인트 파일 삭제
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("체크포인트 파일을 삭제했습니다.")

if __name__ == "__main__":
    main() 