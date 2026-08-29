from dataclasses import dataclass
from typing import Optional

@dataclass
class Receipt:
    """Class for keeping track of a receipt or an invoice."""
    id: str
    image_path: str
    company: Optional[str]
    date: Optional[str]
    address: Optional[str]
    total: Optional[str]


@dataclass
class OCRResult:
    """Class that returns the text extracted from an image + the time taken."""
    id: str
    image_txt: str
    time: float