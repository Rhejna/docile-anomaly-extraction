from torch.utils.data import DataLoader

from src.data.loader import load_receipts
from src.model.dataset import SROIEDataset, split_receipts
from src.model.layoutlm_inputs import load_processor

# ---------------------------------------------------------------------------
# 1. Load Receipts and Split into Train / Validation
# ---------------------------------------------------------------------------
receipts = load_receipts("data/raw/sroie/task2train")
train_receipts, val_receipts = split_receipts(receipts)

print(f"Total receipts loaded : {len(receipts)}")
print(f"Train split size      : {len(train_receipts)}")
print(f"Validation split size : {len(val_receipts)}")


# ---------------------------------------------------------------------------
# 2. Initialize SROIEDataset for Training and Validation
# ---------------------------------------------------------------------------
processor = load_processor()
annotations_dir = "data/raw/sroie/task1train"

train_dataset = SROIEDataset(
    receipts=train_receipts,
    annotations=annotations_dir,
    processor=processor,
)

val_dataset = SROIEDataset(
    receipts=val_receipts,
    annotations=annotations_dir,
    processor=processor,
)

print(f"Train Dataset length  : {len(train_dataset)}")


# ---------------------------------------------------------------------------
# 3. Inspect Tensor Shapes for First Item (dataset[0])
# ---------------------------------------------------------------------------
first_item = train_dataset[0]

print("\n--- First Item Tensor Shapes ---")
for key, tensor in first_item.items():
    print(f"  - {key:<15}: {list(tensor.shape)}")

# Validations
assert list(first_item["input_ids"].shape) == [512], "Expected input_ids shape to be [512]"
assert list(first_item["pixel_values"].shape) == [3, 224, 224], "Expected pixel_values shape to be [3, 224, 224]"


# ---------------------------------------------------------------------------
# 4. Check Last 20 Label Values (Padding Check)
# ---------------------------------------------------------------------------
print("\n--- Last 20 Labels Check ---")
last_20_labels = first_item["labels"][-20:].tolist()
print(f"Last 20 label values: {last_20_labels}")

all_padded = all(label == -100 for label in last_20_labels)
print(f"Are all last 20 labels equal to -100? {all_padded}")

assert all_padded, "Expected all padded labels at the end to be -100!"


# ---------------------------------------------------------------------------
# 5. Create PyTorch DataLoaders (batch_size=4)
# ---------------------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# ---------------------------------------------------------------------------
# 6. Final Test: Inspect First Batch Shapes
# ---------------------------------------------------------------------------
print("\n--- Final Test: DataLoaders Batch Inspection ---")

for first_batch in train_loader:
    print(f"Batch keys: {list(first_batch.keys())}")
    print(f"input_ids shape: {first_batch['input_ids'].shape}")
    print(f"bbox shape     : {first_batch['bbox'].shape}")
    print(f"labels shape   : {first_batch['labels'].shape}")
    print(f"pixel_values   : {first_batch['pixel_values'].shape}")

    # Verify batch size dimension [4, 512]
    assert first_batch["input_ids"].shape == (
        4,
        512,
    ), "Expected input_ids shape to be torch.Size([4, 512])"
    print("\nBatch dimensions successfully validated!")

    break