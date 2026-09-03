import json
import re
from typing import Optional
from dateutil import parser

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


def extract_company(text: str) -> Optional[str]:
    # Preprocessing: Clean empty lines and whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return None

    # Business entity indicators (Malaysia/international)
    company_keywords = r"\b(SDN\b|BHD\b|ENTERPRISE\b|TRADING\b|PERNIAGAAN\b|LIMITED\b|LTD\b|INC\b|CORP\b|MART\b|STORE\b|SHOP\b|KEDAI\b|RESTORAN\b)"

    # (Priority): Search top 5 lines for a business keyword
    max_search_depth = min(5, len(lines))
    for i in range(max_search_depth):
        if re.search(company_keywords, lines[i], re.IGNORECASE):
            return lines[i]

    # (Fallback): Select the first candidate line with sufficient length
    for line in lines:
        clean_len = len(line.replace(" ", ""))
        if clean_len > 10:
            return line

    # (Safe Net): If all lines are short, return the very first non-empty line
    return lines[0]


def extract_address(
    text: str, company_name: Optional[str] = None
) -> Optional[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return None

    # Determine start_index based on provided company_name parameter
    start_index = 0
    if company_name and company_name in lines:
        start_index = lines.index(company_name) + 1
    else:
        start_index = 1 if len(lines) > 1 else 0

    reg_code_pattern = (
        r"(?i)(\bCO\.?\s*REG\b|^\(?[A-Z0-9]{5,12}(?:-[A-Z0-9]{1,3})?\)?$)"
    )
    phone_pattern = r"(\bTEL\b|\bFAX\b|\bPHONE\b|\bTEL\/FAX\b|\+?\d{2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b)"
    date_pattern = r"\b\d{2}[-./]\d{2}[-./]\d{2,4}\b"
    metadata_pattern = (
        r"(~?INVOICE~?|TAX|RECEIPT|CASHIER|OPERATOR|GST|SUMMARY|CHOPPING)"
    )

    address_lines = []

    for line in lines[start_index:]:
        if (
            re.search(phone_pattern, line, re.IGNORECASE)
            or re.search(date_pattern, line, re.IGNORECASE)
            or re.search(metadata_pattern, line, re.IGNORECASE)
        ):
            break

        if re.search(reg_code_pattern, line):
            continue

        address_lines.append(line)

    if address_lines:
        return ", ".join(address_lines)

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
        print(f"Extracted \ntotal: {total}\ndate: {date}\ncompany: {company}\naddress: {address}")

    # --- Run on a small set of files folder ---
    selected_files = ["X00016469622", "X00016469669", "X00016469672", "X51005200938", "X51005255805"]
    for i in  selected_files:
        with open(f"data/interim/ocr/{i}.json", "r", encoding="utf-8") as extraction:
                data = json.load(extraction)["text"]
                total = extract_total(data)
                date = extract_date(data)
                company = extract_company(data)
                address = extract_address(data)
                print(f"\nExtracted \n- total: {total}\n- date: {date}\n- company: {company}\n- address: {address}")

    # --- Full run ---

# To run
# python -m src.ocr.extraction
