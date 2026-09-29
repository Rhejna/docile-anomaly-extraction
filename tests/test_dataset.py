import random

from src.data.loader import load_receipts
from src.model.dataset import split_receipts


receipts = load_receipts("data/raw/sroie/task2train")
train, val = split_receipts(receipts)

# ---------------------------------------------------------------------------
# 1. Tests for split_receipts
# ---------------------------------------------------------------------------
def test_split_receipts_total_count():
    """ Verify that the sum of the two lists is indeed 626."""
    assert len(train) + len(val) == 626


def test_split_receipts_no_overlap():
    """ Verify that no receipt ID appears in both lists."""
    set_id_train = set([x.id for x in train])
    id_val = [x.id for x in val]

    assert set_id_train.isdisjoint(id_val) == True, f"Matching receipts in training and val datasets: {[item for item in id_val if item in set_id_train]}"


def test_split_receipts_reproducible():
    """ Verify that two calls with the same seed produce exactly the same two lists."""
    copied_receipts = receipts.copy()
        
    random.seed(42)
    random.shuffle(copied_receipts)
    split_index = int(len(receipts) * 0.9)
    training = copied_receipts[:split_index]
    validation = copied_receipts[split_index:]

    assert (train, val) == (training, validation)


# To test:
# python -m pytest tests/test_dataset.py -v