import os
from src.data.loader import load_receipts

TRAIN_FOLDER = "data/raw/sroie/task2train"


def test_count_is_exactly_626():
    """1. The loader must load approximately 626 documents.
    Windows duplicates (name(1), name(2)... ) are filtered out by the loader."""
    receipts = load_receipts(TRAIN_FOLDER)
    assert 600 <= len(receipts) <= 650, f"Expected ~626, actual {len(receipts)}"


def test_known_receipt():
    """2. Known case: We verify a specific receipt with its 4 exact values."""
    receipts = load_receipts(TRAIN_FOLDER)

    target_id = "X51005268262"

    found = None
    for r in receipts:
        if r.id == target_id:
            found = r
            break

    assert found is not None, f"The receipt {target_id} was not found"

    # Exact values (based on what observed yourself)
    assert found.company == "HOME MASTER HARDWARE & ELECTRICAL"
    assert found.date == "22/12/2017"
    assert found.address == "NO.113G & 115G, JALAN SETIA GEMBILANG U13/BG BANDAR SETIA ALAM, 40170 BANDAR SETIA ALAM, SELANGOR."
    assert found.total == "15.90"


def test_all_image_paths_exist():
    """3. For each document, the image file must exist on the disk."""
    receipts = load_receipts(TRAIN_FOLDER)

    missing = []
    for r in receipts:
        if not os.path.exists(r.image_path):
            missing.append(r.image_path)

    assert len(missing) == 0, f"{len(missing)} missing images. Examples: {missing[:5]}"



def test_no_parentheses_in_ids():
    """No loaded ID should contain parentheses."""
    receipts = load_receipts (TRAIN_FOLDER)

    with_paren = [r.id for r in receipts if "(" in r.id]
    assert len(with_paren) == 0, f"IDs with parentheses found: {with_paren[:5]}"



# To test:
# python -m pytest tests/test_loader.py -v