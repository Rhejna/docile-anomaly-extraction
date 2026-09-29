import random
from transformers import LayoutLMv3Processor
from torch.utils.data import Dataset

from src.data.labeling import label_segments, read_segments, segments_to_words
from src.data.loader import load_receipts
from src.data.schemas import Receipt
from src.model.layoutlm_inputs import encode_receipt, load_processor


class SROIEDataset(Dataset):

    def __init__(self, receipts: list[Receipt], annotations: str, processor: LayoutLMv3Processor):
        self.receipts = receipts
        self.annotations = annotations
        self.processor = processor

    def __len__(self) -> int :
        return len(self.receipts)

    def __getitem__(self, idx: int) -> dict:
        # 1. Retrieve receipt object directly by integer index from self.receipts
        receipt = self.receipts[idx]

        # 2. Build the path to its annotation file using receipt.id
        file_path = f"{self.annotations}/{receipt.id}.txt"
        segments = read_segments(file_path)

        # 3. Label segments and convert to word-level segments
        labeledsegments, _, _ = label_segments(segments, receipt)
        word_segments = segments_to_words(labeledsegments)

        # 4. Encode receipt with truncation and max_length padding
        encoding = encode_receipt(
            self.processor,
            receipt.image_path,
            word_segments,
            truncation=True,
            padding="max_length"
        )

        # 5. Remove batch dimension [1, ...] using .squeeze(0) on each tensor
        return {key: value.squeeze(0) for key, value in encoding.items()}


def split_receipts(receipts: list[Receipt]) -> tuple[list[Receipt], list[Receipt]]:
    split_index = int(len(receipts) * 0.9)
    copied_receipts = receipts.copy()

    # Set seed for reproducibility and shuffle in-place
    random.seed(42)
    random.shuffle(copied_receipts)

    # Slice train (90%) and validation (10%) sets
    training = copied_receipts[:split_index]
    val = copied_receipts[split_index:]

    return training, val
