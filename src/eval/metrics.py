import csv
import json
from datetime import datetime
from pathlib import Path

from src.text_utils import normalize
from src.data.schemas import Receipt, ExtractedFields

def is_correct(truth: str | None, pred: str | None) -> bool:
    """Return True if the prediction matches the ground truth after normalization.
    
    Cases:
    - both None          → correct (nothing to find, nothing found)
    - truth exists, pred is None → incorrect (missed)
    - values differ      → incorrect
    """
    return normalize(truth) == normalize(pred)


def compare_one(receipt: Receipt, prediction: ExtractedFields) -> dict:
    """Compare one receipt against its prediction, field by field."""
    return {
        "id": receipt.id,
        "company": "success" if is_correct(receipt.company, prediction.company) else "echec",
        "date":    "success" if is_correct(receipt.date, prediction.date) else "echec",
        "address": "success" if is_correct(receipt.address, prediction.address) else "echec",
        "total":   "success" if is_correct(receipt.total, prediction.total) else "echec",
    }


def compute_scores(results: list[dict]) -> dict:
    score_total = 0
    score_date = 0
    score_company = 0
    score_address = 0
    score_global = 0

    for res in results:
        if res["total"] == "success":
            score_total += 1
        if res["address"] == "success":
            score_address += 1        
        if res["company"] == "success":
            score_company += 1
        if res["date"] == "success":
            score_date += 1

    score_total = round((score_total/len(results)) * 100, 1)
    score_date = round((score_date/len(results)) * 100, 1)
    score_company = round((score_company/len(results)) * 100, 1)
    score_address = round((score_address/len(results)) * 100, 1)
    score_global = round((score_total + score_date + score_company + score_address) / 4, 1)

    score = {"company": score_company,
            "date": score_date,
            "address": score_address,
            "total": score_total,
            "global": score_global}

    return score


def save_evaluation_results(method_name: str, detailed_results: list, scores: dict, num_samples: int):
    """Save detailed JSON results and append summary scores to a shared historical CSV log."""
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d")
    detail_path = results_dir / f"{method_name}_{run_date}.json"

    # 1. Save detailed JSON
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {detail_path}")

    # 2. Append to historical scores CSV
    score_log = results_dir / "scores_log.csv"
    file_exists = score_log.exists()

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        method_name,
        scores.get("company", 0.0),
        scores.get("date", 0.0),
        scores.get("address", 0.0),
        scores.get("total", 0.0),
        scores.get("global", 0.0),
        num_samples,
    ]

    headers = [
        "timestamp",
        "method",
        "company",
        "date",
        "address",
        "total",
        "global",
        "size",
    ]

    with open(score_log, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"Scores log updated: {score_log}")