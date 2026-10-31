import argparse
import json
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.loader import load_receipts
from src.model.dataset import SROIEDataset, split_receipts
from src.model.layoutlm_inputs import load_processor
from src.model.model import get_device, load_model
from src.model.train import train

# ---------------------------------------------------------------------------
# Hyperparameters (easy to tweak for experiments)
# ---------------------------------------------------------------------------
EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
CHECKPOINT_DIR = "data/checkpoints/best_model"
RESULTS_DIR = "data/results"


def parse_args():
    """Parse command-line arguments to optionally limit the dataset size."""
    parser = argparse.ArgumentParser(description="Run LayoutLMv3 SROIE training pipeline.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of receipts to use (useful for local debugging).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("--- 1. Loading data ---")
    receipts = load_receipts("data/raw/sroie/task2train")
    
    # Apply limit if specified (e.g., for local testing)
    if args.limit:
        receipts = receipts[:args.limit]
        print(f"Dataset limited to {args.limit} receipts.")

    train_receipts, val_receipts = split_receipts(receipts)
    annotations_dir = "data/raw/sroie/task1train"

    print(f"Train samples: {len(train_receipts)} | Val samples: {len(val_receipts)}")

    print("--- 2. Initializing processor and datasets ---")
    processor = load_processor()

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("--- 3. Setting up model, optimizer, and device ---")
    device = get_device()
    model = load_model()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("--- 4. Starting training loop ---")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    print("--- 5. Saving training history ---")
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results_file = Path(RESULTS_DIR) / "training_history.json"
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print(f"Training history successfully saved to {results_file}")


if __name__ == "__main__":
    main()


# Local : python -m src.model.run_training --limit 8
# Colab : python -m src.model.run_training