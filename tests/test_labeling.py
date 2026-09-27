from pathlib import Path
import pytest

from src.data.schemas import LabeledSegment
from src.text_utils import compact
from src.data.loader import load_receipt, load_receipts
from src.data.labeling import read_segments, label_segments, segments_to_words


# ---------------------------------------------------------------------------
# 1. Tests for text_utils.compact
# ---------------------------------------------------------------------------
def test_compact_none():
    """Ensure compact returns None when input is None."""
    assert compact(None) is None


def test_compact_string():
    """Ensure compact normalizes spaces, casing, and removes punctuation."""
    assert compact("RM 9.00") == "rm900"
    assert compact("  TOTAL : $15.90  ") == "total1590"


# ---------------------------------------------------------------------------
# 2. Tests for read_segments
# ---------------------------------------------------------------------------
def test_read_segments_known_file():
    """
    Test read_segments on a known annotation file.
    Validates total lines and checks hardcoded expected values for line 1.
    """
    annotation_path = Path("data/raw/sroie/task1train/X51005268262.txt")
    assert annotation_path.exists(), f"File {annotation_path} not found for unit test"

    segments = read_segments(str(annotation_path))

    # Check that segment count matches actual file lines (41 lines in X51005268262.txt)
    with open(annotation_path, "r", encoding="utf-8") as f:
        file_lines = [line for line in f if line.strip()]
    assert len(segments) == len(file_lines)

    # Line 1 in file: 146,47,565,47,565,84,146,84,HOME MASTER HARDWARE &
    # Hardcoded expected values calculated independently:
    # x_coords: [146, 565, 565, 146] -> x_min=146, x_max=565
    # y_coords: [47, 47, 84, 84]     -> y_min=47,  y_max=84
    expected_text = "HOME MASTER HARDWARE &"
    expected_pos = (146, 47, 565, 84)

    first_text, first_pos = segments[0]
    assert first_text == expected_text
    assert first_pos == expected_pos


# ---------------------------------------------------------------------------
# 3. Tests for label_segments (known receipt X51005268262)
# ---------------------------------------------------------------------------
def test_label_segments_known_receipt():
    """Test that B-COMPANY and B-TOTAL are correctly assigned to target segments."""
    receipt_id = "X51005268262"
    annotation_path = Path(f"data/raw/sroie/task1train/{receipt_id}.txt")
    
    receipt = load_receipt("data/raw/sroie/task2train", receipt_id)
    segments = read_segments(str(annotation_path))

    labeled_segments, _, _ = label_segments(segments, receipt)

    # Verify HOME MASTER HARDWARE gets labeled as B-COMPANY
    company_match = any(
        "HOME MASTER HARDWARE" in seg.text and seg.label == "B-COMPANY"
        for seg in labeled_segments
    )
    assert company_match, "HOME MASTER HARDWARE should be labeled as B-COMPANY"

    # Verify 15.90 gets labeled as B-TOTAL
    total_match = any(
        "15.90" in seg.text and seg.label == "B-TOTAL"
        for seg in labeled_segments
    )
    assert total_match, "15.90 should be labeled as B-TOTAL"


# ---------------------------------------------------------------------------
# 4. BIO sequence rule test across all receipts (accumulating errors)
# ---------------------------------------------------------------------------
def test_bio_sequence_validity_all_receipts():
    """
    Ensure BIO sequence validity across all receipts:
    An 'I-X' label must NEVER appear without being directly preceded by 'B-X' or 'I-X'.
    Accumulates invalid cases to report all failing receipts at once.
    """
    receipts = load_receipts("data/raw/sroie/task2train")
    annotation_folder = Path("data/raw/sroie/task1train")

    invalid_sequences = []

    for receipt in receipts:
        annotation_path = annotation_folder / f"{receipt.id}.txt"
        if not annotation_path.exists():
            continue

        segments = read_segments(str(annotation_path))
        labeled_segments, _, _ = label_segments(segments, receipt)

        prev_label = "O"
        for idx, seg in enumerate(labeled_segments):
            curr_label = seg.label

            if curr_label.startswith("I-"):
                entity_type = curr_label.split("-")[1]
                expected_prev_b = f"B-{entity_type}"
                expected_prev_i = f"I-{entity_type}"

                if prev_label not in (expected_prev_b, expected_prev_i):
                    invalid_sequences.append(
                        f"Receipt '{receipt.id}' at segment {idx} ('{seg.text}'): "
                        f"found '{curr_label}' preceded by '{prev_label}'"
                    )

            prev_label = curr_label

    assert not invalid_sequences, (
        f"Found {len(invalid_sequences)} BIO sequence violation(s):\n"
        + "\n".join(invalid_sequences)
    )


# ---------------------------------------------------------------------------
# 5. Tests for segments_to_words
# ---------------------------------------------------------------------------
def test_segments_to_words_rules_and_positions():
    """
    Verify transformation of line segments into individual word segments:
    - Correct word splitting per segment.
    - Application of BIO labeling rules (B-X becomes I-X for subsequent words).
    - Inheritance of original segment bounding box (position).
    """
    # 1. Prepare mock input of 3 LabeledSegments
    pos1 = (10, 20, 100, 40)
    pos2 = (10, 45, 100, 60)
    pos3 = (10, 65, 50, 80)

    input_segments = [
        LabeledSegment(text="TAN WOON YANN", position=pos1, label="B-COMPANY"),
        LabeledSegment(text="SDN BHD", position=pos2, label="I-COMPANY"),
        LabeledSegment(text="TOTAL", position=pos3, label="O"),
    ]

    # 2. Run segment-to-words transformation
    word_segments = segments_to_words(input_segments)

    # 3. Assertions
    # Must yield exactly 6 word segments (3 + 2 + 1)
    assert len(word_segments) == 6

    # Verify labels sequence
    expected_labels = [
        "B-COMPANY",
        "I-COMPANY",
        "I-COMPANY",
        "I-COMPANY",
        "I-COMPANY",
        "O",
    ]
    actual_labels = [w.label for w in word_segments]
    assert actual_labels == expected_labels

    # Verify word texts
    expected_texts = ["TAN", "WOON", "YANN", "SDN", "BHD", "TOTAL"]
    actual_texts = [w.text for w in word_segments]
    assert actual_texts == expected_texts

    # Verify position inheritance
    expected_positions = [pos1, pos1, pos1, pos2, pos2, pos3]
    actual_positions = [w.position for w in word_segments]
    assert actual_positions == expected_positions


# python -m pytest tests/test_labeling.py -v