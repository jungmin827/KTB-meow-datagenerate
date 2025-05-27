import json

def extract_captions(json_file_path: str) -> list:
    """
    JSON 파일에서 caption 필드만 추출하는 함수
    
    Args:
        json_file_path: JSON 파일 경로
        
    Returns:
        추출된 caption 목록
    """
    captions = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for item in data:
            if 'topPosts' in item:
                for post in item['topPosts']:
                    if 'caption' in post:
                        captions.append(post['caption'])
    
    return captions

# 사용 예시
file_path = '/Users/jungmin/KTB_teamproject/데이터증강/데이터/0524인스타스크랩3.json'  # 파일 경로 수정
captions = extract_captions(file_path)

# 결과 확인 (처음 몇 개만)
for i, caption in enumerate(captions[:5]):
    print(f"Caption {i+1}: {caption[:100]}...")  # 긴 캡션은 앞부분만 출력

# 결과 다른 파일로 저장
with open('0524captions_only2.json', 'w', encoding='utf-8') as f:  # 출력 파일명 수정
    json.dump(captions, f, ensure_ascii=False, indent=2)