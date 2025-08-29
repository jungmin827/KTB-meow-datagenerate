from KoBERTScore.score import BERTScore
from transformers import AutoTokenizer
import json
import os

class KobertEvaluator:
    def __init__(self, model_name: str = "beomi/kcbert-base", best_layer: int = 4, max_tokens: int = 290):
        if not isinstance(best_layer, int):
            raise ValueError("best_layer는 반드시 int여야 합니다.")
        self.bertscore = BERTScore(model_name, best_layer=best_layer)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def truncate_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > self.max_tokens:
            input_ids = input_ids[:self.max_tokens]
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def evaluate(self, input_path: str, batch_size: int = 128, save_path: str = None) -> float:
        results = []
        data_list = []
        candidates = []
        references = []
        bar_length = 40

        filename = os.path.basename(input_path)
        is_instruct = "instruct" in filename

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if is_instruct:
                    src = self.truncate_text(data.get("input", ""))
                    hyp = self.truncate_text(data.get("output", ""))
                else:
                    src = self.truncate_text(data.get("content", ""))
                    hyp = self.truncate_text(data.get("transformed_content", ""))
                candidates.append(hyp)
                references.append(src)
                data_list.append(data)

        print(f"⭕ 총 {len(data_list)}개 데이터, BERTScore 계산 시작...")
        print(f"-> 데이터셋 : {filename}")

        scores = self.bertscore(references, candidates, batch_size=batch_size)
        if isinstance(scores, tuple) and len(scores) == 3:
            _, _, scores = scores  # F1만 사용

        f1_list = []
        for idx, (data, f1) in enumerate(zip(data_list, scores), 1):
            f1_val = round(float(f1), 5)
            data["kobertscore_f1"] = f1_val
            results.append(data)
            f1_list.append(f1_val)
            percent = idx / len(data_list)
            filled_len = int(bar_length * percent)
            bar = "|" * filled_len + "-" * (bar_length - filled_len)
            print(f"\r진행률: |{bar}| {idx}/{len(data_list)} ({percent*100:.1f}%)", end="", flush=True)
        print()

        # 결과 저장(옵션)
        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                for r in results:
                    json.dump(r, f, ensure_ascii=False)
                    f.write("\n")

        return results

if __name__ == "__main__":
    # ======= 파일 경로 세팅 =======
    input_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/dataset/_dataset/_made/test_made.jsonl"
    output_path = "/Users/seo/Documents/_code/for_AI/my_project/Finetuning/model_eval/_temp/test_made_kobertscore.jsonl"

    evaluator = KobertEvaluator()
    results = evaluator.evaluate(input_path=input_path, batch_size=128, save_path=output_path)

    # ======= 평균 F1 출력 =======
    if results:
        mean_f1 = round(sum([r["kobertscore_f1"] for r in results]) / len(results), 5)
        print(f"평균 KoBERTScore F1: {mean_f1}")
    else:
        print("평가할 데이터가 없습니다.")
