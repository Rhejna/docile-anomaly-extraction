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