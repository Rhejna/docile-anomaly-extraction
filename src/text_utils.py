import re


def compact(value: str | None) -> str | None:
    normalized = normalize(value)
    if normalized is None:
        return None
    return re.sub(r'[^A-Za-z0-9]', '', normalized)


def normalize(value: str | None) -> str | None:
    """Light normalization before comparison.
    - None stays None
    - Extra spaces are removed
    - Case is ignored
    """
    if value is None:
        return None
    return " ".join(value.split()).lower()


if __name__ == "__main__":
    if compact("12/02/2018") in compact("DATE: 12/02/2018"):
        print("well it's inside I guess")

    address = "27,JALAN DEDAP 13, " + "TAMAN JOHOR JAYA," + "81100 JOHOR BAHRU,JOHOR."
    if compact("27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU, JOHOR.") in compact(address):
        print("matched")

# To run
# python -m src.text_utils