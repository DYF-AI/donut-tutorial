
import os
import json
import datasets
from PIL import Image
import numpy as np
from datasets import load_dataset
from datasets.arrow_writer import ArrowWriter
from datasets import Dataset, DatasetDict

dataset_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "image": datasets.features.Image(),
        "text": datasets.Value("string"),
    }
)

def load_image(image_path:str, size=None):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if size is not None:
        # resize image
        image = image.resize((size, size))
        image = np.asarray(image)
        image = image[:, :, ::-1]  # flip color channels from RGB to BGR
        image = image.transpose(2, 0, 1)  # move channels to first dimension
    return image, (w, h)

def generate_example(metadata_path:str, image_path:str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            print(sample)
            file_name = sample["file_name"]
            image_file = os.path.join(image_path, file_name)
            original_image, size = load_image(image_file)
            record = {
                "id": file_name,
                "image": original_image,
                "text": sample["text"],
            }
            yield file_name, record


def build_data(metadata_path:str, image_path:str, output_path:str):
    writer = ArrowWriter(features=dataset_features,
                         path=output_path,
                         # hash_salt="zh"
                         )
    it = generate_example(metadata_path, image_path)
    try:
        for key, record in it:
            example = dataset_features.encode_example(record)
            writer.write(example, key)
    finally:
        num_examples, num_bytes = writer.finalize()
        writer.close()



if __name__ == "__main__":
    target = "/home/dongyongfei786/.cache/huggingface/datasets/imagefolder/default-b31393f5b88f92ce/0.0.0/e872d3ec27c6c200a8881a4af52930df7eca3372b19aa4d0f5db74a2fded8141/cache-98e5d9c22f4e1d67.arrow"
    target_dataset = Dataset.from_file(target)
    print(target_dataset)

    metadata_path = "/mnt/j/dataset/document-intelligence/ICDAR-2019-SROIE-master/data/img/metadata.jsonl"
    image_path = "/mnt/j/dataset/document-intelligence/ICDAR-2019-SROIE-master/data/img"
    output_path = "/mnt/j/dataset/document-intelligence/ICDAR-2019-SROIE-master/data/img/data.arrow"

    build_data(metadata_path, image_path, output_path)

    arrow_dataset = Dataset.from_file(output_path)

    print(arrow_dataset)