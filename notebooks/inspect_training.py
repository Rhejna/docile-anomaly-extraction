from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.loader import load_receipts
from src.model.dataset import SROIEDataset, split_receipts
from src.model.layoutlm_inputs import load_processor
from src.model.model import load_model, get_device
from src.model.train import train

# ---------------------------------------------------------------------------
# 1. Demonstrate that the loop is running and reducing the loss
# ---------------------------------------------------------------------------
receipts = load_receipts("data/raw/sroie/task2train")
train_receipts, val_receipts = split_receipts(receipts)

processor = load_processor()
annotations_dir = "data/raw/sroie/task1train"

train_dataset = SROIEDataset(
    receipts=train_receipts[:8],
    annotations=annotations_dir,
    processor=processor,
)
val_dataset = SROIEDataset(
    receipts=val_receipts[:4],
    annotations=annotations_dir,
    processor=processor,
)

train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)

model = load_model()
optimizer = AdamW(model.parameters(), lr=5e-5)
device = get_device()

checkpoint_dir = "data/checkpoints/test_train_ckpt"

epochs = 10
print(f"Starting training for {epochs} epochs...")
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=epochs,
    checkpoint_dir=checkpoint_dir
)

print("\n--- Test validation ---")

# Check 1: The history list has 10 entries
assert len(history) == epochs, f"Error: The history should contain {epochs} entries, but contains {len(history)}."
print(f"[OK] The history does indeed contain {len(history)} entries.")

# Check 2: The checkpoint directory exists and contains files
ckpt_path = Path(checkpoint_dir)
assert ckpt_path.exists() and ckpt_path.is_dir(), f"Error: The directory {checkpoint_dir} does not exist."

ckpt_files = list(ckpt_path.iterdir())
assert len(ckpt_files) > 0, f"Error: The checkpoint directory {checkpoint_dir} is empty."
print(f"[OK] The checkpoint directory exists and contains {len(ckpt_files)} files: {[f.name for f in ckpt_files]}")

print("\nAll local tests of the train function have passed successfully!")

# To run
# python -m notebooks.inspect_training