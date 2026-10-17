import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from src.data.labeling import label_segments, read_segments, segments_to_words
from src.data.schemas import ExtractedFields, Receipt
from src.eval.metrics import compare_one, compute_scores
from src.model.decode import decode_predictions
from src.model.layoutlm_inputs import encode_receipt


def evaluate_fields(
    model: LayoutLMv3ForTokenClassification, 
    val_receipts: list[Receipt], 
    processor: LayoutLMv3Processor, 
    annotation_folder_path: str
) -> dict:

    comparison_results = []
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

        prediction = ExtractedFields(
            company=decoder["COMPANY"],
            date=decoder["DATE"],
            address=decoder["ADDRESS"],
            total=decoder["TOTAL"],
        )

        comparison = compare_one(receipt, prediction)
        comparison_results.append(comparison)

    return compute_scores(comparison_results)


if __name__ == "__main__":
    from src.data.loader import load_receipts
    from src.model.dataset import split_receipts
    from src.model.layoutlm_inputs import load_processor
    from src.model.model import load_model

    # 1. Load data and split receipts
    receipts = load_receipts("data/raw/sroie/task2train")
    _, val_receipts = split_receipts(receipts)

    # 2. Load model, processor and config
    model = load_model()
    processor = load_processor()
    annotations_dir = "data/raw/sroie/task1train"

    # 3. Test harness on a mini validation subset (5 receipts)
    print("Running evaluation harness test on 5 receipts...")
    scores = evaluate_fields(
        model=model,
        val_receipts=val_receipts[:5],
        processor=processor,
        annotation_folder_path=annotations_dir
    )

    print("\n--- Evaluation Test Scores ---")
    for field, score in scores.items():
        print(f"{field:10}: {score:.2f}%")