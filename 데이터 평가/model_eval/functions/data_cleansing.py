import re
import json

def get_emoji_pattern():
    return re.compile(
        "[" +
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]", flags=re.UNICODE
    )

def get_allowed_char_pattern():
    allowed = (
        r'가-힣'
        r'a-zA-Z0-9'
        r' .,!?~'
        r'🐾😢🔥😤❓❤️🧡💛💚💙💜🖤🤍🤎'
        r'🐕🐩🐈🐈‍⬛🐱🐶😹😺😸😻😼😽😾😿🫨'
        r'😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗🤩'
        r'🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫🥱😴😌😛😜😝🤤'
        r'😒😓😔😕🙃🤑😲☹️🙁😖😞😟😤😢😭😦😧😨😩🤯😬😰😱😳🥺😵🥶🥵😡😠🤬😷🤒🤕🤢🤮🥵🥶🥴🤧'
        r'🐻🦊🐼🐷🐮🐸🐵🐔🦄🦁🐯🐴🦓🦍🐧🦆🦉🦇🦜🦋'
        r'🍖🍗🍕🍔🍟🌭🍿🍩🍪🍫🍬🍭🍡🍨🍧🍦🍤🍣🍚🍙🍘🥚🥞🥯🥐🥖🍞🥨'
        r'❤️🧡💛💚💙💜🖤🤍🤎💔💕💞💓💗💖💘💝💟'
        r'💤💢💦💧💫💥💬💭🗯️✨⭐🌟🔥🌈☁️⛈️❄️🌤️🌙☀️'
    )
    return re.compile(rf'[^{allowed}]')

def clean_text(text):
    # 1. rn, \r\n, \n, \r → 공백
    text = re.sub(r"(\\r\\n|\\n|\\r|rn)", " ", text)
    # 2. URL 제거
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # 3. 허용 문자 이외 삭제
    text = get_allowed_char_pattern().sub("", text)
    # 4. 동일 문자 4회 이상 반복 → 2회, 동일 이모지 4회 이상 반복 → 3회
    text = re.sub(r'(\S{1,5})\1{3,}', r'\1\1', text)
    text = re.sub(r'((?:' + get_emoji_pattern().pattern + r')){4,}', lambda m: m.group(0)[:3], text)
    # 5. 다중 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_jsonl_replace_fields(
    input_path,
    output_path,
    fields=("content", "transformed_content")
):
    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            data = json.loads(line)
            for field in fields:
                if field in data and isinstance(data[field], str):
                    data[field] = clean_text(data[field])
            # 둘 중 하나라도 빈 문자열이면 저장하지 않음
            if any(data.get(field, "").strip() == "" for field in fields):
                continue
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            written += 1
    print(f"[클랜징] content, transformed_content 둘 다 빈 값이 아닌 {written}개만 저장 완료")

if __name__ == "__main__":
    input_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_made/dataset_0709_made.jsonl"
    output_path = "/Users/jaeseoksee/Documents/project/for_AI/my_project/Finetuning/dataset/_dataset/_filtered/dataset_0709_made_clean.jsonl"
    clean_jsonl_replace_fields(input_path, output_path)
