import json
import re
from typing import Optional
from dateutil import parser

# def extract_total(text) -> Optional[str]:
#     pattern = r"(?:\d+\.\d{2}|\d+\,\d{2})"
#     list_text = text.split("\n")
#     cash_change = []

#     for subtext in list_text:
#         cash = re.search("CASH", subtext, re.IGNORECASE)
#         change = re.search("CHANGE", subtext, re.IGNORECASE)
#         if cash or change:
#             cash_change.append(re.search(pattern, subtext))
    
#     for subtext in list_text:
#         match = re.search("total", subtext, re.IGNORECASE)
#         if match:
#             total = re.search(pattern, subtext)
#             if total:
#                 if abs(float(cash_change[0].group().replace(',', '.')) - float(cash_change[1].group().replace(',', '.'))) ==  float(total.group().replace(',', '.')):
#                     print(total.group().replace(',', '.'))
#                     return total.group().replace(',', '.')
#             print(match.group())
#     return None


def extract_total(text: str) -> Optional[str]:
    # Regex pattern to capture amounts with 2 decimal places (e.g., 33.90, 33,90)
    pattern = r"\d+[\.\,]\d{2}"
    list_text = text.split("\n")

    cash_val: Optional[float] = None
    change_val: Optional[float] = None

    # 1. Safe extraction of CASH and CHANGE values
    for subtext in list_text:
        if re.search(r"\bcash\b", subtext, re.IGNORECASE):
            match = re.search(pattern, subtext)
            if match:
                cash_val = float(match.group().replace(",", "."))

        if re.search(r"\bchange\b", subtext, re.IGNORECASE):
            match = re.search(pattern, subtext)
            if match:
                change_val = float(match.group().replace(",", "."))

    # 2. Total extraction and cross-validation
    for subtext in list_text:
        # print(subtext)
        if re.search(r"\btotal\b", subtext, re.IGNORECASE):
            match_total = re.search(pattern, subtext)
            if match_total:
                total_val = float(match_total.group().replace(",", "."))

                # Primary strategy: Validate using CASH - CHANGE
                if cash_val is not None and change_val is not None:
                    expected_total = abs(cash_val - change_val)

                    # Floating-point safe comparison (tolerance check)
                    if abs(expected_total - total_val) < 0.01:
                        return f"{total_val:.2f}"

    return None


def extract_date(text: str) -> Optional[str]:
    # Regex pattern to capture common date formats (e.g., DD/MM/YYYY, DD-MM-YY, DD.MM.YYYY)
    pattern = r"\b\d{2}[-./]\d{2}[-./]\d{2,4}\b"
    lines = text.split("\n")

    # Helper function to validate if a string represents a valid date
    def is_valid_date(date_str: str) -> bool:
        try:
            parser.parse(date_str)
            return True
        except (ValueError, TypeError, parser.ParserError):
            return False

    # Strategy 1 (Priority): Search for lines containing the keyword "date"
    for line in lines:
        if re.search(r"\bdate\b", line, re.IGNORECASE):
            match = re.search(pattern, line)
            if match:
                extracted_value = match.group()
                if is_valid_date(extracted_value):
                    return extracted_value

    # Strategy 2 (Fallback): Full text search using original logic
    match = re.search(pattern, text)
    if match:
        extracted_value = match.group()
        if is_valid_date(extracted_value):
            return extracted_value

    # Default return if no valid date is found
    return None


def extract_company(text) -> Optional[str]:
    # Preprocessing / Line splitting
    lines = text.split("\n")

    # 2. Target search with safety checks
    for line in lines:
        # Target regex pattern
        match = re.search(r"YOUR_PATTERN", line, re.IGNORECASE)
        if match:
            # Value extraction and domain validation
            extracted_value = match.group()
            if is_valid(extracted_value):  # e.g., date validity check
                return format_value(extracted_value)

    # 3. Fallback strategy or explicit None return
    return None

def extract_address(text) -> Optional[str]:
    # 1. Preprocessing / Line splitting
    lines = text.split("\n")

    # 2. Target search with safety checks
    for line in lines:
        # Target regex pattern
        match = re.search(r"YOUR_PATTERN", line, re.IGNORECASE)
        if match:
            # Value extraction and domain validation
            extracted_value = match.group()
            if is_valid(extracted_value):  # e.g., date validity check
                return format_value(extracted_value)

    # 3. Fallback strategy or explicit None return
    return None



if __name__ == "__main__":
    # --- Test on 1 file ---
    file = "data/interim/ocr/X00016469620.json"
    with open(file, "r", encoding="utf-8") as extraction:
        data = json.load(extraction)["text"]
        total = extract_total(data)
        date = extract_date(data)
        company = extract_company(data)
        address = extract_address(data)
        print("Extracted total:", total)
        print("Extracted date:", date)
        print("Extracted date:", company)
        print("Extracted date:", address)

    # --- Run on a small set of files folder ---
    # selected_files = ["X00016469622", "X00016469669", "X00016469672", "X51005200938", "X51005255805"]
    # for i in  selected_files:
    #     with open(f"data/interim/ocr/{i}.json", "r", encoding="utf-8") as extraction:
    #             data = json.load(extraction)["text"]
    #             total = extract_total(data)
    #             date = extract_date(data)
    #             # print("Extracted total:", total)
    #             print("Extracted date:", date)


    # --- Full run ---
    # texte = "quelque chose avec \nTotal ici".split("\n")
    # print(texte)
    # x = re.search("total", texte, re.IGNORECASE)
    # print(x)

# To run
# python -m src.ocr.extraction
