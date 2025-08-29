import re
import json

class QualityEvaluator:
    def __init__(self):
        self.min_length = 6
        self.max_length = 100
        self.max_repeat = 3
        self.forbidden_words = ['씨발', 'ㅅㅂ', 'fuck', 'shit', '좆', '병신']
        self.emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        self.allowed_pattern = re.compile(r'^[가-힣a-zA-Z0-9 .,!?~🐾😢🔥😤❓]*$')

    def has_forbidden_word(self, text: str) -> float:
        return 0.0 if any(word in text for word in self.forbidden_words) else 1.0

    def repeat_score(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        max_count = max([words.count(w) for w in set(words)])
        if max_count <= self.max_repeat:
            return 1.0
        return max(0.0, 1.0 - (max_count - self.max_repeat) / 7.0)

    def emoji_score(self, text: str) -> float:
        return 1.0 if self.emoji_pattern.search(text) else 0.5

    def length_score(self, text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        l = len(text)
        if self.min_length <= l <= self.max_length:
            return 1.0
        if l < self.min_length:
            return l / self.min_length
        if l > self.max_length:
            return max(0.0, 1.0 - (l - self.max_length) / self.max_length)
        return 0.0

    def natural_score(self, text: str) -> float:
        if not text:
            return 0.0
        allowed = self.allowed_pattern.findall(text)
        ratio = len(''.join(allowed)) / len(text) if len(text) > 0 else 0
        return min(1.0, max(0.0, ratio))

    def overlap_score(self, src: str, hyp: str) -> float:
        src_set = set(src.split())
        hyp_set = set(hyp.split())
        if not src_set or not hyp_set:
            return 0.0
        overlap = len(src_set & hyp_set) / len(src_set | hyp_set)
        return min(1.0, overlap / 0.2) if overlap < 0.2 else 1.0

    def evaluate(self, input_path: str) -> list:
        results = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                src = data.get("content", "")
                hyp = data.get("transformed_content", "")

                if not isinstance(hyp, str):
                    total_score = 0.0
                else:
                    scores = [
                        self.length_score(hyp),
                        self.has_forbidden_word(hyp),
                        self.repeat_score(hyp),
                        self.natural_score(hyp),
                        self.emoji_score(hyp),
                        self.overlap_score(src, hyp)
                    ]
                    total_score = sum(scores) / len(scores)

                results.append({
                    "quality_score": round(total_score, 3)
                })
        return results