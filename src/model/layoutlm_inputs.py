from PIL import Image
from transformers import AutoProcessor, BatchEncoding, LayoutLMv3Processor
from src.data.labeling import label_segments, read_segments, segments_to_words
from src.data.loader import load_receipt
from src.data.schemas import LabeledSegment
from src.data.labels import label2id

MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LENGTH = 512

def _clip_1000(val: int) -> int:
    """Helper function to constrain an integer value between 0 and 1000."""
    return max(0, min(1000, val))

def normalize_box(position: tuple[float, float, float, float], width: float, height: float) -> tuple[int, int, int, int]:
    """
    Normalizes a bounding box (x_min, y_min, x_max, y_max) to an integer 0-1000 scale.
    """
    x_min_raw, y_min_raw, x_max_raw, y_max_raw = position

    x_min = _clip_1000(int((x_min_raw / width) * 1000))
    y_min = _clip_1000(int((y_min_raw / height) * 1000))
    x_max = _clip_1000(int((x_max_raw / width) * 1000))
    y_max = _clip_1000(int((y_max_raw / height) * 1000))

    return  (x_min, y_min, x_max, y_max)


def load_processor() -> LayoutLMv3Processor :
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
    return processor


def encode_receipt(
    processor: LayoutLMv3Processor, img_path: str, 
    segments: list[LabeledSegment], truncation: bool = False,
    padding: bool | str = False, max_length: int = 512) -> BatchEncoding :
    with Image.open(img_path) as im:
        image = im.convert("RGB")
        size = image.size
        text_list = [seg.text for seg in segments]
        boxes_list = [normalize_box(seg.position, size[0], size[1]) for seg in segments]
        labels_list = [label2id[seg.label] for seg in segments]

    return processor(images=image, text=text_list, 
                     boxes=boxes_list, word_labels=labels_list, return_tensors="pt", 
                     truncation=truncation, padding=padding, max_length=max_length)


# To run
# python -m src.model.layoutlm_inputs