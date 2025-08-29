import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Optional

class BleuEvaluator:
    def __init__(self, ref_key: str = "content", hyp_key: str = "transformed_content"):
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def calc_bleu(self, reference: str, hypothesis: str) -> float:
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        smoothie = SmoothingFunction().method4
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        return round(score, 5)

    def evaluate_jsonl(
        self,
        jsonl_path: str,
        output_path: Optional[str] = None
    ) -> List[Optional[float]]:
        bleu_scores: List[Optional[float]] = []
        output_data: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                reference = data.get(self.ref_key, "")
                hypothesis = data.get(self.hyp_key, "")
                if reference and hypothesis:
                    bleu = self.calc_bleu(reference, hypothesis)
                    bleu_score = min(bleu * 10.0, 1.0)  # 정규화
                    bleu_scores.append(bleu_score)
                    data["bleu"] = bleu
                    data["bleu_score"] = bleu_score
                else:
                    bleu_scores.append(None)
                    data["bleu"] = None
                    data["bleu_score"] = None
                output_data.append(data)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as f:
                for item in output_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return bleu_scores

if __name__ == "__main__":
    jsonl_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/_made/dataset_0629_made.jsonl"
    output_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/model_eval/_temp/dataset_0629_made_bleu.jsonl"
    evaluator = BleuEvaluator()
    bleu_scores = evaluator.evaluate_jsonl(jsonl_path, output_path)
    bleu_valid = [b for b in bleu_scores if b is not None]
    if bleu_valid:
        avg_bleu = sum(bleu_valid) / len(bleu_valid)
        print(f"전체 샘플 수: {len(bleu_valid)}")
        print(f"BLEU 평균 점수: {avg_bleu:.5f}")
    else:
        print("평가할 데이터가 없습니다.")
