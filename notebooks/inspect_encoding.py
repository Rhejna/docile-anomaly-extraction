from pathlib import Path

from src.data.labeling import label_segments, read_segments, segments_to_words
from src.data.loader import load_receipt, load_receipts
from src.model.layoutlm_inputs import load_processor, encode_receipt
from src.data.labels import id2label


receipt_id = "X51005268262"
task1_path = f"data/raw/sroie/task1train/{receipt_id}.txt"
task2_path = "data/raw/sroie/task2train"

raw_segments = read_segments(task1_path)
receipt = load_receipt(task2_path, receipt_id)
labeledsegments, _, _ = label_segments(raw_segments, receipt)
word_segments = segments_to_words(labeledsegments)

processor = load_processor()
encoding = encode_receipt(processor, receipt.image_path, word_segments)


# ---------------------------------------------------------------------------
# Verification 1: Tensor Keys and Shapes
# ---------------------------------------------------------------------------
print("\n--- Verification 1: Keys and Shapes ---")
print(f"Type of object : {type(encoding)}")
print(f"Keys           : {list(encoding.keys())}")
print("Shapes:")
for key, tensor in encoding.items():
    print(f"  - {key}: {tensor.shape}")

# Validate sequence lengths alignment
input_len = encoding["input_ids"].shape[1]
bbox_len = encoding["bbox"].shape[1]
labels_len = encoding["labels"].shape[1]
attention_len = encoding["attention_mask"].shape[1]

assert (
    input_len == bbox_len == labels_len == attention_len
), "Sequence length mismatch across text-based tensors!"


# ---------------------------------------------------------------------------
# Verification 2: Token-to-Label Alignment
# ---------------------------------------------------------------------------
print("\n--- Verification 2: Token-to-Label Alignment ---")

# Convert input_ids tensor to readable token strings
input_ids_list = encoding["input_ids"][0].tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(input_ids_list)

labels_list = encoding["labels"][0].tolist()
word_ids_list = encoding.word_ids(batch_index=0)

print(f"{'Index':<6} | {'Word ID':<8} | {'Token':<20} | {'Label ID':<8} | {'Readable Label'}")
print("-" * 65)

for idx, (token, label_id, word_id) in enumerate(
    zip(tokens, labels_list, word_ids_list)
):
    # Decode integer label ID back to human-readable string or -100
    readable_label = id2label.get(label_id, "-100") if label_id != -100 else "-100"
    word_id_str = str(word_id) if word_id is not None else "None"

    print(
        f"{idx:<6} | {word_id_str:<8} | {token:<20} | {label_id:<8} | {readable_label}"
    )


# ---------------------------------------------------------------------------
# Verification 3: Sequence Length Statistics Across All Receipts
# ---------------------------------------------------------------------------
print("\n--- Verification 3: Sequence Length Check (All Receipts) ---")

all_receipts = load_receipts(task2_path)
task1_dir = Path("data/raw/sroie/task1train")

total_receipts = 0
over_512_count = 0
max_length_observed = 0

print(f"Processing sequence lengths for {len(all_receipts)} receipts...")

for rec in all_receipts:
    ann_path = task1_dir / f"{rec.id}.txt"
    if not ann_path.exists():
        continue

    rec_segments = read_segments(str(ann_path))
    rec_labeled, _, _ = label_segments(rec_segments, rec)
    rec_words = segments_to_words(rec_labeled)

    rec_encoding = encode_receipt(processor, rec.image_path, rec_words)
    seq_len = rec_encoding["input_ids"].shape[1]

    total_receipts += 1
    if seq_len > max_length_observed:
        max_length_observed = seq_len

    if seq_len > 512:
        over_512_count += 1

print(f"Total processed receipts : {total_receipts}")
print(f"Max token length found   : {max_length_observed}")
print(f"Receipts exceeding 512   : {over_512_count}")


# ---------------------------------------------------------------------------
# Verification 4: Bounding Box Coordinates Range Safety
# ---------------------------------------------------------------------------
print("\n--- Verification 4: Bounding Box Range Check ---")

bbox_tensor = encoding["bbox"]
min_bbox_val = bbox_tensor.min().item()
max_bbox_val = bbox_tensor.max().item()

print(f"Min bbox value : {min_bbox_val} (Expected >= 0)")
print(f"Max bbox value : {max_bbox_val} (Expected <= 1000)")

assert min_bbox_val >= 0, "Error: Found negative coordinate in bounding boxes!"
assert max_bbox_val <= 1000, "Error: Found coordinate exceeding 1000 in bounding boxes!"
print("Bounding box coordinate values are strictly within [0, 1000].")

# To run
# python -m notebooks.inspect_encoding