from pathlib import Path
from transformers import LayoutLMv3ForTokenClassification

from src.model.model import get_device


def save_checkpoint(model, output_dir: str) -> None:
    """Saves the fine-tuned model and its configuration to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

def load_checkpoint(checkpoint_dir: str) -> LayoutLMv3ForTokenClassification:
    """Loads a fine-tuned model checkpoint from disk and moves it to the active device."""
    device = get_device()
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_dir)
    return model.to(device)