import os
import pickle
import json
import random
import shutil
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

TRAIN_IMAGE_DP = "/mnt/j/dataset/document-intelligence/EATEN数据集/dataset_trainticket/train/hcp_aug_2"
TEST_IMAGE_DP = "/mnt/j/dataset/document-intelligence/EATEN数据集/dataset_trainticket/test"
# LABEL_DP include train and test images
LABEL_DP = "/mnt/j/dataset/document-intelligence/EATEN数据集/dataset_trainticket/real_1920.pkl"

def generate_matadata(image_root, label_path):
    """
        {"file_name": "0001.png", "text": "This is a golden retriever playing with a ball"}
        {"file_name": "0002.png", "text": "A german shepherd"}
    """
    f = open(label_path, "rb")
    label_data = pickle.load(f)
    # define metadata list
    metadata_list = []
    files = [os.path.splitext(file)[0] for file in os.listdir(image_root)]
    # parse metadata
    for file_name in label_data.keys():
        if file_name not in files:
            continue
        sample_data = label_data[file_name]
        # create "text" column with json string
        text = json.dumps(sample_data, ensure_ascii=False)
        # add to metadata list if image exists
        if os.path.exists(os.path.join(image_root, file_name+".jpg")):
            metadata_list.append({"text":text, "file_name":f"{file_name}.jpg"})
        elif os.path.exists(os.path.join(image_root, file_name+".JPG")):
            metadata_list.append({"text": text, "file_name": f"{file_name}.JPG"})
    # write jsonline file
    # In the same path as the picture
    with open(os.path.join(image_root, "metadata.jsonl"), "w") as jsonl_file:
        for metadata in metadata_list:
            json.dump(metadata, jsonl_file, ensure_ascii=False)
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
    generate_matadata(TRAIN_IMAGE_DP, LABEL_DP)
    generate_matadata(TEST_IMAGE_DP, LABEL_DP)
    #load(DP)
