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
    
