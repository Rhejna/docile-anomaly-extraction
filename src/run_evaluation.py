from src.data.loader import load_receipts
from src.data.schemas import ExtractedFields
from src.eval.metrics import compare_one, compute_scores, save_evaluation_results
from src.model.dataset import split_receipts
from src.extract.extraction import extract_company, extract_date, extract_address, extract_total
from src.ocr.ocr import get_text_from_img


def main():
    folder_path = "data/raw/sroie/task2train"
    method_name = "baseline_regex"

    # 1. Load data and use split_receipts to evaluate strictly on the 63 validation receipts
    receipts = load_receipts(folder_path)
    _, val_receipts = split_receipts(receipts)

    print(f"Running baseline evaluation on {len(val_receipts)} validation receipts...")

    detailed_results = []   # Detailed predictions for the JSON output
    comparison_results = [] # Evaluation status for compute_scores

    for receipt in val_receipts:
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

        # Record detailed record per receipt
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

    # Compute overall field scores
    scores = compute_scores(comparison_results)

    # Save detailed JSON and update the central CSV log via the shared utility function
    save_evaluation_results(
        method_name=method_name,
        detailed_results=detailed_results,
        scores=scores,
        num_samples=len(val_receipts),
    )


if __name__ == "__main__":
    main()

# To run
# python -m src.run_evaluation