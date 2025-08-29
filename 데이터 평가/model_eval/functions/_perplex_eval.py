import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from typing import List

class PerplexityEvaluator:
    def __init__(
        self,
        model_name: str = "skt/kogpt2-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def calculate_ppl(self, sentence: str) -> float:
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    @staticmethod
    def score_by_threshold(ppl: float) -> float:
        if 60 <= ppl <= 180:
            return 1.0
        elif 35 <= ppl < 60 or 180 < ppl <= 350:
            return 0.8
        elif 20 <= ppl < 35 or 350 < ppl <= 700:
            return 0.6
        elif 700 < ppl <= 2000:
            return 0.4
        else:
            return 0.2

    def calculate_perplexity_batch(
        self,
        input_jsonl_path: str,
        output_jsonl_path: str = None,
        field: str = "transformed_content",
        batch_size: int = 16,
    ) -> list:
        """
        파일로부터 읽어서 배치 평가 (기존 방식)
        """
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            datas = [json.loads(line) for line in f]

        score_list = []

        for i in tqdm(range(0, len(datas), batch_size), desc="Evaluating"):
            batch = datas[i:i + batch_size]
            for j, data in enumerate(batch):
                text = data.get(field, "")
                try:
                    ppl = self.calculate_ppl(text)
                    score = self.score_by_threshold(ppl)
                except Exception:
                    ppl = -1.0
                    score = 0.0
                datas[i + j]["raw_perplexity"] = ppl
                datas[i + j]["perplexity_score"] = score
                score_list.append(score)

        if output_jsonl_path is not None:
            with open(output_jsonl_path, "w", encoding="utf-8") as f:
                for data in datas:
                    json.dump(data, f, ensure_ascii=False)
                    f.write("\n")

        return score_list

    def calc_perplexity_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> list:
        """
        텍스트 리스트로 배치 평가 (main.py에서 별도 래핑 필요 없음)
        """
        score_list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    ppl = self.calculate_ppl(text)
                    score = self.score_by_threshold(ppl)
                except Exception:
                    score = 0.0
                score_list.append(score)
        return score_list

if __name__ == "__main__":
    input_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/_made/test_made.jsonl"
    output_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/model_eval/_temp/test_made_perplexity.jsonl"

    evaluator = PerplexityEvaluator()
    # 파일 전체 배치 평가 (파일 I/O)
    scores = evaluator.calculate_perplexity_batch(
        input_jsonl_path=input_path,
        output_jsonl_path=output_path,
        field="transformed_content",
        batch_size=32
    )
    # 텍스트 리스트 평가 (리스트로 바로)
    # 예시: scores = evaluator.calculate_perplexity_batch_from_texts(["문장1", "문장2"], batch_size=8)
    valid_scores = [s for s in scores if s > 0]
    avg_score = round(sum(valid_scores) / len(valid_scores), 5) if valid_scores else 0.0
    print(f"평균 Perplexity 점수: {avg_score}")
