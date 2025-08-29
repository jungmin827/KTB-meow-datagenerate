import csv
import os
import json
from _kobert_eval import KobertEvaluator
from _type_eval import TypeEvaluator
from _quality_eval import QualityEvaluator
from _perplex_eval import PerplexityEvaluator
from _bleu_eval import BleuEvaluator
from typing import Optional, List, Dict
import argparse

import warnings
warnings.filterwarnings("ignore")

def print_eval_stats(results, prefix=""):
    thres_koberscore =  0.6
    thres_typescore = 0.8
    thres_qualityscore = 0.8
    thres_bleu = 0.4
    thres_perplexity = 0.5
    
    if not results:
        print(f"{prefix}데이터 없음")
        return
    if "kobertscore_f1" in results[0]:
        scores = [float(r["kobertscore_f1"]) for r in results if "kobertscore_f1" in r]
        mean_score = sum(scores) / len(scores)
        below_thres = sum(1 for s in scores if s < thres_koberscore)
        total = len(scores)
        print(f"⭐ kobertscore_f1 평균: {mean_score:.3f} (bad-data count : {below_thres}개 / {total}개)")
    if "type_score" in results[0]:
        scores = [float(r["type_score"]) for r in results if r["type_score"] is not None]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        below_thres = sum(1 for s in scores if s < thres_typescore)
        total = len(scores)
        print(f"⭐ type_score 평균: {mean_score:.3f} (bad-data count : {below_thres}개 / {total}개)")
    if "quality_score" in results[0]:
        scores = [float(r["quality_score"]) for r in results if r["quality_score"] is not None]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        below_thres = sum(1 for s in scores if s < thres_qualityscore)
        total = len(scores)
        print(f"⭐ quality_score 평균: {mean_score:.3f} (bad-data count : {below_thres}개 / {total}개)")
    if "bleu_score" in results[0]:
        scores = [float(r["bleu_score"]) for r in results if r.get("bleu_score") is not None]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        below_thres = sum(1 for s in scores if s < thres_bleu)
        total = len(scores)
        print(f"⭐ bleu_score 평균: {mean_score:.3f} (bad-data count : {below_thres}개 / {total}개)")
    if "perplexity_score" in results[0]:
        scores = [float(r["perplexity_score"]) for r in results if r.get("perplexity_score") is not None]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        below_thres = sum(1 for s in scores if s < thres_perplexity)
        total = len(scores)
        print(f"⭐ perplexity_score 평균: {mean_score:.3f} (bad-data count : {below_thres}개 / {total}개)")

def run_all_evals(
    input_path: str,
    use_kobert: bool = False,
    use_type: bool = False,
    use_quality: bool = False,
    use_bleu: bool = False,
    use_perplexity: bool = False,
    output_path: str = None
):
    with open(input_path, "r", encoding="utf-8") as f:
        original_data = [json.loads(line) for line in f]

    n = len(original_data)
    results: List[Dict] = [{} for _ in range(n)]

    # KoBERTScore
    if use_kobert:
        kobert_eval = KobertEvaluator(model_name="beomi/kcbert-base", best_layer=4)
        kbs_results = kobert_eval.evaluate(input_path)
        print_eval_stats(kbs_results)
        for i, r in enumerate(kbs_results):
            results[i]["kobertscore_f1"] = r.get("kobertscore_f1")

    # Type Score
    if use_type:
        type_eval = TypeEvaluator()
        type_results = type_eval.evaluate(input_path)
        print_eval_stats(type_results)
        for i, r in enumerate(type_results):
            results[i]["type_score"] = r.get("type_score")

    # Quality Score
    if use_quality:
        quality_eval = QualityEvaluator()
        quality_results = quality_eval.evaluate(input_path)
        print_eval_stats(quality_results)
        for i, r in enumerate(quality_results):
            results[i]["quality_score"] = r.get("quality_score")

    # BLEU Score (BleuEvaluator에서 점수만 받아옴)
    if use_bleu:
        bleu_evaluator = BleuEvaluator()
        bleu_scores = bleu_evaluator.evaluate_jsonl(input_path, output_path=None)
        for i, bleu in enumerate(bleu_scores):
            results[i]["bleu_score"] = bleu
        bleu_valid = [b for b in bleu_scores if b is not None]
        if bleu_valid:
            avg_bleu = sum(bleu_valid) / len(bleu_valid)
            below_thres = sum(1 for b in bleu_valid if b < 0.02)
            print(f"⭐ bleu_score 평균: {avg_bleu:.3f} (bad-data count : {below_thres}개 / {len(bleu_valid)}개)")

    # Perplexity Score
    if use_perplexity:
        texts = [orig.get("transformed_content", "") for orig in original_data]
        evaluator = PerplexityEvaluator(model_name="skt/kogpt2-base-v2")
        perplexity_scores = evaluator.calc_perplexity_batch(
            texts,
            batch_size=8,
        )
        for i, score in enumerate(perplexity_scores):
            results[i]["perplexity_score"] = score
        print_eval_stats([{"perplexity_score": r.get("perplexity_score")} for r in results])


    # 통합 저장
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            for orig, res in zip(original_data, results):
                merged = orig.copy()
                merged.update(res)
                f.write(json.dumps(merged, ensure_ascii=False) + "\n")
        print(f"✅ 최종 통합 평가 결과 저장: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_kobert", action="store_true")
    parser.add_argument("--use_type", action="store_true")
    parser.add_argument("--use_quality", action="store_true")
    parser.add_argument("--use_bleu", action="store_true")
    parser.add_argument("--use_perplexity", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    run_all_evals(
        input_path=args.input_path,
        use_kobert=args.use_kobert,
        use_type=args.use_type,
        use_quality=args.use_quality,
        use_bleu=args.use_bleu,
        use_perplexity=args.use_perplexity,
        output_path=args.output_path
    )