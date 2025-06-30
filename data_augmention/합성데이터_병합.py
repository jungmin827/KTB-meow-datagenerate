import os
import json
from pathlib import Path
from typing import List, Generator
import logging
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_temp_files(data_dir: str) -> List[Path]:
    """데이터 디렉토리에서 모든 미야옹_대량_합성_데이터.jsonl.temp_* 파일을 찾습니다."""
    data_path = Path(data_dir)
    return sorted(list(data_path.glob("미야옹_대량_합성_데이터.jsonl.temp_*")))

def count_lines(file_path: Path) -> int:
    """파일의 라인 수를 계산합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def read_temp_file(file_path: Path) -> Generator[dict, None, None]:
    """미야옹_대량_합성_데이터.jsonl.temp_* 파일을 한 줄씩 읽어서 JSON 객체로 변환합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON in {file_path}: {e}")
                continue

def merge_temp_files(input_dir: str, output_file: str) -> None:
    """모든 미야옹_대량_합성_데이터.jsonl.temp_* 파일을 하나의 JSONL 파일로 병합합니다."""
    # 입력 파일 목록 가져오기
    input_files = get_temp_files(input_dir)
    if not input_files:
        logger.error(f"No temp files found in {input_dir}")
        return

    logger.info(f"Found {len(input_files)} temp files to merge")
    
    # 전체 라인 수 계산
    total_lines = sum(count_lines(f) for f in input_files)
    logger.info(f"Total lines to process: {total_lines:,}")

    # 출력 파일 생성
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # tqdm으로 진행 상황 표시
        with tqdm(total=total_lines, desc="Merging files") as pbar:
            for input_file in input_files:
                logger.info(f"Processing {input_file}")
                for json_obj in read_temp_file(input_file):
                    outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    pbar.update(1)

    logger.info(f"Successfully merged all files into {output_file}")

if __name__ == "__main__":
    # 설정
    DATA_DIR = "데이터"  # 데이터 디렉토리
    OUTPUT_FILE = "데이터/merged_miyang_data.jsonl"  # 출력 파일 경로

    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 파일 병합 실행
    merge_temp_files(DATA_DIR, OUTPUT_FILE) 