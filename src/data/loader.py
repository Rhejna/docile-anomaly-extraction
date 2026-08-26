import json
import os
import re
from .schemas import Receipt

def load_receipts(folder_path: str) -> list:

    all_files = []
    unique_names = set()
    files_ignored = []
    duplicates_discarded = 0
    receipt_list = []
    valid_extensions = (".jpg", ".txt")

    # Lists all the answer files
    for x in os.listdir(folder_path) :
        if x.lower().endswith(valid_extensions):
            all_files.append(x)
            name_without_ext = os.path.splitext(x)[0]

            # Ignore Windows duplicates such as name(1), name(2)..
            if re.search(r"\(\d+\)$", name_without_ext):
                duplicates_discarded += 1
                continue
            
            unique_names.add(name_without_ext)

    # Finds the corresponding image
    for filename in unique_names:
        txt_name = f"{filename}.txt"
        jpg_name = f"{filename}.jpg"

        if txt_name in all_files and jpg_name in all_files:

            # Reads the 4 answers
            txt_path = os.path.join(folder_path, txt_name)
            with open(txt_path, "r", encoding="utf-8") as annotation:
                data = json.load(annotation)

                # Creates an object and adds it to a list
                receipt_info = Receipt(
                    id= filename,
                    image_path= os.path.join(folder_path, jpg_name),
                    company= data.get("company"),
                    date= data.get("date"),
                    address= data.get("address"),
                    total= data.get("total"),
                )

                receipt_list.append(receipt_info)

        elif txt_name in all_files:
            files_ignored.append(txt_name)
        else:
           files_ignored.append(jpg_name)

    print(f"Charged : {len(receipt_list)}")
    print(f"Duplicates discarded : {duplicates_discarded}")
    print(f"Excluded (imprévus) : {len(files_ignored)}")

    # if files_ignored:
    #     print("Files excluded :")
    #     for f in sorted(files_ignored):
    #         print(" ", f)

    return receipt_list