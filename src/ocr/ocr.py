import json

import pytesseract
from PIL import Image
import time
from src.data.schemas import OCRResult
from src.data.loader import load_receipts
from pathlib import Path
from tqdm import tqdm

def get_text_from_img(image_path: str, force: bool = False) -> OCRResult:
    image_id = Path(image_path).stem
    cache_dir = Path("data/interim/ocr")
    cache_file = cache_dir / f"{image_id}.json"

    # First, check if the cache file exists
    if cache_file.exists() and not force:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return OCRResult(id=image_id, image_txt=data["text"], time=data["time"])

    # Otherwise → perform OCR, save the result, then return
    start = time.perf_counter()

    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image=image, lang="eng")       
     
    elapsed = time.perf_counter() - start
    text_dict = {"text": extracted_text, "time": elapsed}

    cache_dir.mkdir(parents=True, exist_ok=True)  # crée le dossier s'il n'existe pas
    cache_file.write_text(json.dumps(text_dict), encoding="utf-8")

    return OCRResult(id=image_id, image_txt=extracted_text, time=elapsed)


def get_text_from_folder(image_paths: list) -> None:
    fails = 0
    empty_files = 0
    time_files = []
    total_time = 0

    for image_path in tqdm(image_paths, desc="OCR"):
        try:
            file = get_text_from_img(image_path)
        except Exception as e:
            print(f"Failed on {image_path}: {e}")
            fails +=1
            continue
        
        if len(file.image_txt) < 50:
            empty_files +=1

        total_time += file.time
        time_files.append((file.id, file.time))

    # Calcul of the average, median and worst cases
    if time_files:
        sort_by_time = sorted(time_files, key=lambda x: x[1])
        n = len(sort_by_time)
        mid = n // 2
        if n % 2 == 1:
            median_time = sort_by_time[mid][1]
        else:
            median_time = (sort_by_time[mid - 1][1] + sort_by_time[mid][1]) / 2

        avg_time = total_time / n
        worst = sort_by_time[-1][1]
    else:
        median_time = avg_time = worst = 0.0
        sort_by_time = []


    print(f"There are {fails} failures in total")
    print(f"There are {empty_files} empty or nearly empty texts (fewer than 50 characters) in total")
    print(f"Average time: {avg_time:.2f}s. Median time: {median_time:.2f}s. Worst-case time: {worst:.2f}s")
    print(f"Top 5 worst cases: {sort_by_time[-5:]}")


def list_images_in_folder(folder_path: str) -> list:
    folder = Path(folder_path)
    image_paths = [str(p) for p in folder.glob("*.jpg")] + [str(p) for p in folder.glob("*.png")]

    return image_paths


if __name__ == "__main__":
    # --- Test on 1 image ---
    # image_path = "data/raw/sroie/task2train/X00016469612.jpg"
    # result = get_text_from_img(image_path)
    # print(f"Temps : {result.time:.2f}s")
    # print(result.image_txt[:500])

    # --- Run on a small test folder ---
    # list_receipts = list_images_in_folder("data/ocr_test_data")
    # get_text_from_folder(list_receipts)

    # --- Full run ---
    folder_path = "data/raw/sroie/task2train"
    list_receipts = [r.image_path for r in load_receipts(folder_path)]
    get_text_from_folder(list_receipts)


# To run
# python -m src.ocr.ocr