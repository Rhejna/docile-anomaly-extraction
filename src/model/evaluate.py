import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import argparse

from src.data.labeling import label_segments, read_segments, segments_to_words
from src.data.loader import load_receipts
from src.data.schemas import ExtractedFields, Receipt
from src.eval.metrics import compare_one, compute_scores, save_evaluation_results # Function shared with run_evaluation.py
from src.model.checkpoint import load_checkpoint
from src.model.decode import decode_predictions
from src.model.dataset import split_receipts
from src.model.layoutlm_inputs import encode_receipt, load_processor
from src.normalize.postprocess import clean_prediction  # Import your postprocessing function


def evaluate_fields(
    model: LayoutLMv3ForTokenClassification, 
    val_receipts: list[Receipt], 
    processor: LayoutLMv3Processor, 
    annotation_folder_path: str
) -> dict:
    
    raw_comparison_results = []
    cleaned_comparison_results = []
    
    model.eval()
    device = next(model.parameters()).device

    for receipt in val_receipts:
        file_path = f"{annotation_folder_path}/{receipt.id}.txt"
        segments = read_segments(file_path)

        labeledsegments, _, _ = label_segments(segments, receipt)
        word_segments = segments_to_words(labeledsegments)

        encoding = encode_receipt(
            processor=processor,
            img_path=receipt.image_path,
            segments=word_segments,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predicted label IDs as a list of integers
        labels = logits.argmax(dim=-1).squeeze(0).tolist()

        decoder = decode_predictions(
            labels=labels,
            word_ids_list=encoding.word_ids(batch_index=0),
            segments=word_segments
        )

        # 1. Build raw prediction object
        raw_prediction = ExtractedFields(
            company=decoder["COMPANY"],
            date=decoder["DATE"],
            address=decoder["ADDRESS"],
            total=decoder["TOTAL"],
        )

        # 2. Build cleaned prediction object using postprocessing rules
        cleaned_prediction = ExtractedFields(
            company=clean_prediction("company", decoder["COMPANY"]),
            date=clean_prediction("date", decoder["DATE"]),
            address=clean_prediction("address", decoder["ADDRESS"]),
            total=clean_prediction("total", decoder["TOTAL"]),
        )

        # Compare both versions against ground truth
        raw_comparison_results.append(compare_one(receipt, raw_prediction))
        cleaned_comparison_results.append(compare_one(receipt, cleaned_prediction))

    # Compute scores for both approaches
    raw_scores = compute_scores(raw_comparison_results)
    cleaned_scores = compute_scores(cleaned_comparison_results)

    # Return both metrics dicts and detailed results for JSON logging
    return {
        "raw": raw_scores,
        "cleaned": cleaned_scores,
        "detailed": cleaned_comparison_results  # or raw/cleaned details as needed
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LayoutLMv3 SROIE model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/best_model",
        help="Path to the model checkpoint directory.",
    )
    args = parser.parse_args()

    # 1. Load data and split receipts (get all validation samples)
    receipts = load_receipts("data/raw/sroie/task2train")
    _, val_receipts = split_receipts(receipts)

    # 2. Load model from checkpoint and processor
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint)
    processor = load_processor()
    annotations_dir = "data/raw/sroie/task1train"

    # 3. Run evaluation harness on ALL validation receipts
    print(f"Running evaluation on all {len(val_receipts)} validation receipts...")
    results = evaluate_fields(
        model=model,
        val_receipts=val_receipts,  # Evaluating all receipts, not just [:5]
        processor=processor,
        annotation_folder_path=annotations_dir
    )

    # 4. Display comparison scores (Raw vs Cleaned)
    print("\n--- Evaluation Test Scores Comparison ---")
    print(f"{'Field':10} | {'Raw (%)':10} | {'Cleaned (%)':12} | {'Gap (%)':8}")
    print("-" * 50)
    
    for field in ["company", "date", "address", "total"]:
        raw_val = results["raw"].get(field, 0.0)
        cleaned_val = results["cleaned"].get(field, 0.0)
        gap = cleaned_val - raw_val
        print(f"{field:10} | {raw_val:10.2f} | {cleaned_val:12.2f} | {gap:+8.2f}")

    # 5. Save results using the shared utility function (using cleaned scores for the log)
    save_evaluation_results(
        method_name="layoutlmv3_cleaned",
        detailed_results=results.get("detailed", []),
        scores=results["cleaned"],
        num_samples=len(val_receipts)
    )