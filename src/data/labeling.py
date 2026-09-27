from pathlib import Path

from src.data.loader import load_receipt, load_receipts
from src.text_utils import compact
from .schemas import Receipt, LabeledSegment


def read_segments(annotation_path: str) -> list[tuple]:
    segments = []
    ignored_lines = 0

    with open(annotation_path, 'r', encoding='utf-8') as file:
        for line in file:
            # .strip() removes the trailing newline character '\n' and extra whitespaces
            parts = line.strip().split(",", 8)

            if len(parts) != 9:
                ignored_lines += 1
                continue

            # Group the first 8 items back together, and keep the rest as the second item
            try:
                coords = [int(x) for x in parts[:8]]
            except ValueError:
                ignored_lines += 1
                continue
            text = parts[8]

            # some_list[start:stop:step]
            x_min = min(coords[0::2])
            y_min = min(coords[1::2]) 
            x_max = max(coords[0::2]) 
            y_max = max(coords[1::2])

            position = (x_min, y_min, x_max, y_max)
            segments.append((text, position))

    if ignored_lines > 0:
        print(f"⚠️  {ignored_lines} ligne(s) ignorée(s) dans {annotation_path}")

    return segments

def label_segments(segments: list, receipt: Receipt) -> tuple[list[LabeledSegment], dict, dict]:
    labeled_segments = []

    # --- Prepare truths ---
    truths = {
        "company": compact(receipt.company),
        "date": compact(receipt.date),
        "address": compact(receipt.address),
        "total": compact(receipt.total),
    }

    # --- Compact segments once ---
    seg_text = [compact(seg[0]) or "" for seg in segments]
    seg_size = len(seg_text)

    windows = {field: None for field in truths}
    ambiguous = {field: 0 for field in truths}

    MAX_EXTRA = 15

    def find_permissive(truth: str, max_offset: int) -> list[tuple[int, int, int]]:
        """Truth contained in a window of segments (company, date, address)."""
        candidates = []
        for i in range(seg_size):
            comparison = ""
            for offset in range(max_offset):
                if i + offset >= seg_size:
                    break
                comparison += seg_text[i + offset]
                if truth in comparison and len(comparison) <= len(truth) + MAX_EXTRA:
                    candidates.append((i, offset + 1, len(comparison) - len(truth)))
                    break
        return candidates

    def find_total(truth: str) -> list[int]:
        """Single segment equal to truth, or truth preceded by letters only (e.g. 'rm900')."""
        matches = []
        for i, text in enumerate(seg_text):
            if text == truth:
                matches.append(i)
            elif text.endswith(truth) and text[:-len(truth)].isalpha():
                matches.append(i)
        return matches

    # ========== 1. Company, date, address (permissive) ==========
    max_offsets = {"company": 2, "date": 2, "address": 6}

    for entity, max_offset in max_offsets.items():
        truth = truths[entity]
        if truth is None:
            continue

        candidates = find_permissive(truth, max_offset)
        if not candidates:
            continue

        best_extra = min(extra for _, _, extra in candidates)
        best = [c for c in candidates if c[2] == best_extra]
        windows[entity] = (best[0][0], best[0][1])
        ambiguous[entity] = len(best) - 1

    # ========== 2. Total (strict, single segment) ==========
    truth = truths["total"]
    if truth is not None:
        matches = find_total(truth)

        # Prefer matches whose own segment or previous segment mentions "total"
        keyword_matches = [
            i for i in matches
            if "total" in segments[i][0].lower()
            or (i > 0 and "total" in segments[i - 1][0].lower())
        ]

        chosen_pool = keyword_matches or matches
        if chosen_pool:
            windows["total"] = (chosen_pool[-1], 1)   # last occurrence = usually the final total
            ambiguous["total"] = len(chosen_pool) - 1

    # ========== 3. Build the labels ==========
    label_dict = {}
    for entity, values in windows.items():
        if values is None:
            continue
        start_index, length = values
        for win in range(length):
            index = start_index + win
            if index in label_dict:
                continue   # don't overwrite a field already labeled
            label_dict[index] = f"B-{entity.upper()}" if win == 0 else f"I-{entity.upper()}"

    for index, (text, position) in enumerate(segments):
        labeled_segments.append(
            LabeledSegment(text=text, position=position, label=label_dict.get(index, "O"))
        )

    return labeled_segments, windows, ambiguous

def segments_to_words(labeledSegments: list[LabeledSegment])  -> list[LabeledSegment]:
    labeled_words = []
    for seg in labeledSegments:
        text = seg.text.split()
        position = seg.position
        label = seg.label

        if not text: 
            continue

        if label.startswith("B-"):
            labeled_words.append(LabeledSegment(text[0], position, label))
            labeled_words.extend([LabeledSegment(x, position, label.replace("B-", "I-")) for x in text[1:]])
            continue

        labeled_words.extend([LabeledSegment(x, position, label) for x in text])

    return labeled_words

if __name__ == "__main__":
    from collections import defaultdict

    # --- Test on 1 file ---
    file_path = "data/raw/sroie/task1train/X00016469619.txt"
    segments = read_segments(file_path)
    receipt = load_receipt("data/ocr_test_data", "X00016469619")
    labeled_results, windows, ambiguous = label_segments(segments, receipt)
    segments_to_words(label_segments(segments, receipt))
    
    # print()

    # --- Run on a folder ---
    # receipts = load_receipts("data/ocr_test_data")
    # # receipts = load_receipts("data/raw/sroie/task2train")

    # annotation_folder = Path("data/raw/sroie/task1train")

    # total_files = 0
    # none_count = defaultdict(int)
    # ambiguous_count = defaultdict(int)
    # missing_annotation = 0

    # for receipt in receipts:
    #     receipt_id = receipt.id
    #     annotation_path = annotation_folder / f"{receipt_id}.txt"

    #     if not annotation_path.exists():
    #         missing_annotation += 1
    #         continue

    #     segments = read_segments(str(annotation_path))

    #     try:
    #         labeled_results, windows, ambiguous = label_segments(segments, receipt)
    #         total_files += 1

    #         # Agrégation
    #         for field in ["company", "date", "address", "total"]:
    #             if windows[field] is None:
    #                 none_count[field] += 1
    #             ambiguous_count[field] += ambiguous[field]

    #     except Exception as e:
    #         print(f"Error processing {receipt_id}: {e}")

    # # --- Résumé final ---
    # print("\n" + "="*60)
    # print(f"Processed {total_files} files")
    # print(f"Missing annotation files: {missing_annotation}")
    # print("-"*60)
    # print(f"{'Field':<12} {'None (not found)':<20} {'Total Ambiguous'}")
    # print("-"*60)
    # for field in ["company", "date", "address", "total"]:
    #     print(f"{field:<12} {none_count[field]:<20} {ambiguous_count[field]}")
    # print("="*60)



# To run
# python -m src.data.labeling