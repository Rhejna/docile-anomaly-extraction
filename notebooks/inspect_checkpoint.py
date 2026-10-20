import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.loader import load_receipts
from src.model.checkpoint import load_checkpoint, save_checkpoint
from src.model.dataset import SROIEDataset, split_receipts
from src.model.layoutlm_inputs import load_processor
from src.model.model import get_device, load_model
from src.model.train import train_one_epoch

# ---------------------------------------------------------------------------
# 1. Setup small dataset and train for 1 epoch
# ---------------------------------------------------------------------------
receipts = load_receipts("data/raw/sroie/task2train")
train_receipts, _ = split_receipts(receipts)

processor = load_processor()
annotations_dir = "data/raw/sroie/task1train"

train_dataset = SROIEDataset(
    receipts=train_receipts[:8],
    annotations=annotations_dir,
    processor=processor,
)

train_loader = DataLoader(train_dataset, batch_size=2)
device = get_device()

model = load_model()
optimizer = AdamW(model.parameters(), lr=5e-5)

print("Training original model for 1 epoch...")
train_one_epoch(model, train_loader, optimizer, device)

# ---------------------------------------------------------------------------
# 2. Save and reload checkpoint
# ---------------------------------------------------------------------------
checkpoint_dir = "data/checkpoints/test_checkpoint"
print(f"Saving checkpoint to {checkpoint_dir}...")
save_checkpoint(model, checkpoint_dir)

print(f"Loading checkpoint from {checkpoint_dir}...")
reloaded_model = load_checkpoint(checkpoint_dir)

# ---------------------------------------------------------------------------
# 3. Compare logits between original and reloaded models
# ---------------------------------------------------------------------------
model.eval()
reloaded_model.eval()

sample_batch = next(iter(train_loader))
inputs = {k: v.to(device) for k, v in sample_batch.items()}

with torch.no_grad():
    original_logits = model(**inputs).logits
    reloaded_logits = reloaded_model(**inputs).logits

# Check exact tensor equality
tensors_match = torch.allclose(original_logits, reloaded_logits, atol=1e-5)

print("\n--- Checkpoint Verification ---")
if tensors_match:
    print("SUCCESS: Reloaded model logits match the original model exactly!")
else:
    print("FAILURE: Logits mismatch! Checkpoint saving or loading is flawed.")

# To run
# python -m notebooks.inspect_checkpoint