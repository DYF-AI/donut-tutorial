import os
import json
import random
import shutil
from pathlib import Path
from datasets import load_dataset

DP = r"J:\dataset\document-intelligence\ICDAR-2019-SROIE-master\data"

def generate_matadata(data_root):
    """
        {"file_name": "0001.png", "text": "This is a golden retriever playing with a ball"}
        {"file_name": "0002.png", "text": "A german shepherd"}
    """
    base_path = Path(data_root)
    metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")
    # define metadata list
    metadata_list = []

    # parse metadata
    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            # load json file
            json_data = json.load(json_file)
            # create "text" column with json string
            text = json.dumps(json_data)
            # add to metadata list if image exists
            if image_path.joinpath(file_name.stem + ".jpg").is_file():
                metadata_list.append({"text":text, "file_name":f"{file_name.stem}.jpg"})
    # write jsonline file
    # In the same path as the picture
    with open(image_path.joinpath("metadata.jsonl"), "w") as jsonl_file:
        for metadata in metadata_list:
            json.dump(metadata, jsonl_file)
            jsonl_file.write("\n")


def load(data_root):
    base_path = Path(data_root)
    #metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")
    # load dataset
    dataset = load_dataset("imagefolder", data_dir = image_path, split="train")

    print(f"Dataset has {len(dataset)} images")
    print(f"Dataset features are: {dataset.features.keys()}")
    random_sample = random.randint(0, len(dataset))

    print(f"Random sample is {random_sample}")
    print(f"OCR text is {dataset[random_sample]['text']}")


#     OCR text is {"company": "LIM SENG THO HARDWARE TRADING", "date": "29/12/2017", "address": "NO 7, SIMPANG OFF BATU VILLAGE, JALAN IPOH BATU 5, 51200 KUALA LUMPUR MALAYSIA", "total": "6.00"}

if __name__ == "__main__":
    generate_matadata(DP)
    #load(DP)
