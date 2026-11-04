import re


def clean_prediction(field: str, value: str | None) -> str | None:
    """Clean a predicted field value to remove formatting artifacts before comparison.
    
    Args:
        field: The name of the field (e.g., 'date', 'total', 'company', 'address').
        value: The raw predicted string value or None.
        
    Returns:
        The cleaned string value or None.
    """
    if value == None:
        return None

    value = value.strip()
    if not value:
        return None

    if field == "date":
        # Extract standard date patterns (e.g., DD/MM/YYYY, YYYY-MM-DD) to remove time components
        match = re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', value)
        if match:
            return match.group(0)
        # Fallback: take the first whitespace-separated token
        return value.split()[0]

    elif field == "total":
        # Remove currency symbols (like RM, $, €) and extract the numeric price value
        match = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', value.replace(',', ''))
        if match:
            return match.group(0)
        return value

    elif field in ["company", "address"]:
        # Standardize internal whitespace for text fields without altering semantic content
        return re.sub(r'\s+', ' ', value)

    return value