import csv
import json
from datetime import datetime
from pathlib import Path

from src.data.schemas import ExtractedFields
from src.data.loader import load_receipts
from src.eval.metrics import compare_one, compute_scores
from src.extract.extraction import extract_company, extract_date, extract_address, extract_total
from src.ocr.ocr import get_text_from_img


def main():
    # Use the folder that has both images AND annotations
    sub_folder_path = "data/ocr_test_data" # for test on a small subfolder
    folder_path = "data/raw/sroie/task2train"   
    method_name = "baseline_regex"

    receipts = load_receipts(folder_path)

    detailed_results = []   # for the full JSON
    comparison_results = [] # for compute_scores

    for receipt in receipts:
        ocr = get_text_from_img(receipt.image_path)

        company = extract_company(ocr.image_txt)

        prediction = ExtractedFields(
            company=company,
            date=extract_date(ocr.image_txt),
            address=extract_address(ocr.image_txt, company_name=company),
            total=extract_total(ocr.image_txt),
        )

        # Field-by-field comparison
        comparison = compare_one(receipt, prediction)
        comparison_results.append(comparison)

        # Rich detail for later analysis (worst cases, etc.)
        detailed_results.append({
            "id": receipt.id,
            "truth": {
                "company": receipt.company,
                "date": receipt.date,
                "address": receipt.address,
                "total": receipt.total,
            },
            "prediction": {
                "company": prediction.company,
                "date": prediction.date,
                "address": prediction.address,
                "total": prediction.total,
            },
            "correct": {
                "company": comparison["company"] == "success",
                "date": comparison["date"] == "success",
                "address": comparison["address"] == "success",
                "total": comparison["total"] == "success",
            },
        })

    # --- Compute scores ---
    scores = compute_scores(comparison_results)

    # --- 1. Save detailed JSON (one file per run) ---
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d")
    detail_path = results_dir / f"{method_name}_{run_date}.json"

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {detail_path}")

    # --- 2. Append one line to the scores log CSV ---
    score_log = results_dir / "scores_log.csv"
    file_exists = score_log.exists()

    # Prepare the row in the exact column order
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        method_name,
        scores["company"],
        scores["date"],
        scores["address"],
        scores["total"],
        scores["global"],
        len(comparison_results),  # size
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
    print(scores)


if __name__ == "__main__":
    main()

# To run
# python -m src.run_evaluation