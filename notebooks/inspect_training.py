from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.loader import load_receipts
from src.model.dataset import SROIEDataset, split_receipts
from src.model.layoutlm_inputs import load_processor
from src.model.model import load_model, get_device
from src.model.train import train_one_epoch

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

train_loader = DataLoader(train_dataset, batch_size=2)
model = load_model()
optimizer = AdamW(model.parameters(), lr=5e-5)

for i in range(5):
    loss = train_one_epoch(model, train_loader, optimizer, get_device())
    print(f"Mean loss: {loss}")

# To run
# python -m notebooks.inspect_training