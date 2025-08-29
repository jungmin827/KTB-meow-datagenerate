import re
import json
import os

class TypeEvaluator:
    """
    동물 말투(고양이/강아지) 스타일이 잘 반영되었는지 평가
    """

    def __init__(self):
        self.cat_endings = [
            # [기본 및 가장 흔한 패턴]
            '냥', '냐옹', '이냥', '이다냥', '다냥', '냐용', '이냐옹', '다옹', '댜옹',

            # [추가/감정/감탄/강조형]
            '냐하', '먀', '먀하', '냐앙', '냐우', '냐욧',
            '냥냥', '냐옹이', '냐앙앙', '냐야', '냐오', '냐온', '냐홍', '냐뇽', '냐웅', '냐오옹', '먀옹', '먀먀', '냐햏',

            # [신조어/유행/복수/복합/반복형]
            '냔', '뇽', '뇽뇽', '먀옹먀옹', '냐홍이', '냐옹냥', '냐옹냥냥', '냐하하', '냐핫', '먀하하', '냐옹옹',
            '냐아앙', '냐아앙앙', '냥옹', '냔냥', '냐야옹', '냐옹냐옹',

            # [띄어쓰기, 기호 결합/확장(띄어쓰기/기호 포함)]
            ' 냥', ' 냐옹', ' 냐', ' 냐욧', ' 냐옹이', ' 냥냥',
            '~냥', '~냐옹', '~냐', '~냐하', '~냐용', '~먀', '~냐앙',

            # [말 끝/추임새형/기호 포함]
            '먀', '냐옹~', '냐앙~', '냐하~', '냐옹!', '냐앙!', '냐하!', '냐옹.', '냐앙.', '냐하.', '먀.', '먀!', '냐야~', '냐옹냐옹~',

            # [고양이 대표명사/지칭어]
            '집사'
        ]

        self.dog_endings = [
            # [기본]
            '멍', '왈', '다멍', '다개', '다왈', '요멍', '왕', '왕왕', '멍멍', '멍멍이', '컹', '컹컹', '왈왈', '멍이',

            # [복합/반복/확장/강조]
            '멍이멍이', '멍왕', '멍멍멍', '멍왈', '멍컹', '왈멍', '멍컹컹', '멍왈왈',
            '왕멍', '왕왕왕', '컹컹컹', '왈왈왈', '컹멍', '멍컹왈', '왕이', '왈이', '몽', '몽몽', '멍뭉', '왈뭉', '몽왈',
            '멍멍왈', '왕왈', '멍몽', '왈몽', '멍몽왈', '왕몽', '컹컹왕', '왕컹', '왕컹컹',

            # [띄어쓰기, 기호 결합/확장(띄어쓰기/기호 포함)]
            ' 멍', ' 왈', '~멍', '~왈', '~왕', '~멍멍', '~왕왕', '~다멍', '~다개', '~다왈', '~컹', '~멍이', '~멍멍이',

            # [고양이+강아지 혼합 말투(이상치 탐지)]
            '냐멍', '냐왈', '냥왈', '냥멍', '냐멍멍', '냐왕', '냥왕', '냥멍멍',

            # [강아지 대표명사/지칭어]
            '주인'
        ]


        self.cat_nouns = [
            '고양이', '냥이', '야옹이', '캣', '냥냥이', '묘', '묘님', '캣초딩', '캣맘', '냥스타그램', '묘생', '캣타워', '미스코리냥'
        ]
        self.dog_nouns = [
            '강아지', '댕댕이', '멍멍이', '개', '견', '댕댕', '견생', '개스타그램', '멍스타그램', '견주', '멍뭉이', '미스코리냥'
        ]

        self.cat_pattern = re.compile(r"(" + "|".join([re.escape(e) for e in self.cat_endings]) + r")([\W\s]|$)", re.IGNORECASE)
        self.dog_pattern = re.compile(r"(" + "|".join([re.escape(e) for e in self.dog_endings]) + r")([\W\s]|$)", re.IGNORECASE)

        self.cat_noun_pattern = re.compile(r"|".join([re.escape(n) for n in self.cat_nouns]), re.IGNORECASE)
        self.dog_noun_pattern = re.compile(r"|".join([re.escape(n) for n in self.dog_nouns]), re.IGNORECASE)

    def remove_nouns(self, text: str) -> str:
        text = self.cat_noun_pattern.sub(' ', text)
        text = self.dog_noun_pattern.sub(' ', text)
        return text

    def type_score(self, post_type: str, transformed: str) -> float:
        text_wo_noun = self.remove_nouns(transformed)
        has_cat = bool(self.cat_pattern.search(text_wo_noun))
        has_dog = bool(self.dog_pattern.search(text_wo_noun))
        if post_type == "dog":
            if has_cat and not has_dog:
                return 0.1
            elif has_cat and has_dog:
                return 0.2
            elif has_dog:
                return 1.0
            else:
                return 0.0
        elif post_type == "cat":
            if has_dog and not has_cat:
                return 0.1
            elif has_cat and has_dog:
                return 0.2
            elif has_cat:
                return 1.0
            else:
                return 0.0
        else:
            return -1

    def evaluate(self, input_path: str, output_path:str =None) -> float:
        results = []
        total_score = 0.0
        count = 0

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                post_type = data.get("post_type", "")
                transformed = data.get("transformed_content", "")
                score = self.type_score(post_type, transformed)
                data["type_score"] = score
                results.append(data)
                if score is not None:
                    total_score += score
                    count += 1

        # output_path 지정 시만 저장
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

        return results


if __name__ == "__main__":
    input_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/_made/test_made.jsonl"
    output_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/model_eval/_temp/test_made_type.jsonl"

    evaluator = TypeEvaluator()
    results = evaluator.evaluate(input_path=input_path, output_path=output_path)

    # 평균 산출
    type_scores = [d["type_score"] for d in results if d.get("type_score") is not None]
    avg_score = round(sum(type_scores) / len(type_scores), 5) if type_scores else 0.0

    print(f"✅ Type 평가 완료 → 저장 위치: {output_path}")
    print(f"평균 Type Score: {avg_score}")
