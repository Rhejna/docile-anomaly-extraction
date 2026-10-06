import torch
from transformers import LayoutLMv3ForTokenClassification

from src.data.labels import id2label, label2id, LABELS
from .layoutlm_inputs import MODEL_NAME


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> LayoutLMv3ForTokenClassification:
    device = get_device()
    print(f"Using device: {device}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id) 
    return model.to(device)


if __name__ == "__main__":
    load_model()

# To run
# python -m src.model.model