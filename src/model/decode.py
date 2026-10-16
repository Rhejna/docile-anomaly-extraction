from src.data.schemas import LabeledSegment
from src.data.labels import id2label

FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]


def decode_predictions(
    labels: list[int],
    word_ids_list: list[int | None],
    segments: list[LabeledSegment],
) -> dict[str, str | None]:
    """Decodes token-level label predictions back into structured receipt fields.

    Args:
        labels: Predicted class label IDs per token.
        word_ids_list: Mapping from tokens to word indices (from encoding.word_ids()).
        segments: Original word-level segments passed to encode_receipt.

    Returns:
        Dictionary containing predicted strings for 'company', 'date', 'address', 'total'.
    """

    # First token prediction per word_id
    word_predictions: dict[int, int] = {}

    for label_id, word_id in zip(labels, word_ids_list):
        if word_id is not None and word_id not in word_predictions:
            word_predictions[word_id] = label_id

    
    # Convert label IDs to readable strings
    word_labels: list[tuple[str, str]] = []
    for word_id, label_id in sorted(word_predictions.items()):
        word_text = segments[word_id].text
        label_str = id2label[label_id]
        word_labels.append((word_text, label_str))

    
    # Group consecutive words into field segments
    field_words: dict[str, list[str]] = {field: [] for field in FIELDS}
    current_field: str | None = None

    for text, label in word_labels:
        if label == "O":
            current_field = None
            continue

        prefix, field_name = label.split("-", 1) if "-" in label else ("", "")

        if prefix == "B" and field_name in FIELDS:
            # Keep only the first extracted segment per field
            if not field_words[field_name]:
                current_field = field_name
                field_words[field_name].append(text)
            else:
                current_field = None

        elif prefix == "I" and field_name == current_field:
            field_words[field_name].append(text)

        else:
            current_field = None

    
    # join words with spaces or assign None
    return {
        field: " ".join(words) if words else None
        for field, words in field_words.items()
    }


if __name__ == "__main__":
    from src.data.labeling import label_segments, read_segments, segments_to_words
    from src.data.loader import load_receipts
    from src.model.layoutlm_inputs import encode_receipt, load_processor

    # Load a single receipt and its corresponding annotation file
    receipts = load_receipts("data/raw/sroie/task2train")
    receipt = receipts[0]
    annotations_dir = "data/raw/sroie/task1train"

    # Extract word-level labeled segments
    segments = read_segments(f"{annotations_dir}/{receipt.id}.txt")
    labeled_segments, _, _ = label_segments(segments, receipt)
    word_segments = segments_to_words(labeled_segments)

    # Encode receipt inputs
    processor = load_processor()
    encoding = encode_receipt(
        processor,
        receipt.image_path,
        word_segments,
        truncation=True,
        padding="max_length",
    )

    # Extract ground-truth labels as simulated model predictions
    labels_simulated = encoding["labels"].squeeze(0).tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # Decode predictions back into fields
    result = decode_predictions(labels_simulated, word_ids, word_segments)

    print(f"--- Decoded results for receipt {receipt.id} ---")
    for key, val in result.items():
        print(f"{key:8}: {val}")


# To run
# python -m src.model.decode