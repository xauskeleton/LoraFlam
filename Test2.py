from datasets import Dataset
from PIL import Image
import os
import json
import torch
import sys

# Import image_processor từ open_clip và tokenizer từ transformers
try:
    import open_clip
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install with 'pip install open_clip_torch transformers'.")
    sys.exit(1)

# Kiểm tra phiên bản torch (>=2.1 cho open_flamingo tương thích)
if torch.__version__ < "2.1":
    print(
        f"Warning: PyTorch version {torch.__version__} is < 2.1. OpenFlamingo requires >= 2.1. Please upgrade with 'pip install --upgrade torch torchvision'.")
    sys.exit(1)

# Load image_processor (từ OpenFlamingo 3B config: ViT-L/14 openai)
model_clip, _, image_processor = open_clip.create_model_and_transforms('ViT-L/14', pretrained='openai')

# Load tokenizer (từ OpenFlamingo 3B config: mpt-1b-redpajama-200b)
tokenizer = AutoTokenizer.from_pretrained("anas-awadalla/mpt-1b-redpajama-200b")

# Set pad_token to eos_token if not defined (fixes the padding error)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if __name__ == '__main__':
    # Đường dẫn thư mục dataset
    data_dir = "MedPix-2-0"
    images_dir = os.path.join(data_dir, "images")

    # Kiểm tra và load Descriptions.json
    descriptions_path = os.path.join(data_dir, "Descriptions.json")
    if not os.path.exists(descriptions_path):
        raise FileNotFoundError(f"Descriptions.json not found in {data_dir}")
    with open(descriptions_path, "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    # Load Case_topic.json
    case_topic_path = os.path.join(data_dir, "Case_topic.json")
    if not os.path.exists(case_topic_path):
        raise FileNotFoundError(f"Case_topic.json not found in {data_dir}")
    with open(case_topic_path, "r", encoding="utf-8") as f:
        case_topics = json.load(f)

    # Tạo ánh xạ U_id -> Case Diagnosis
    case_diagnosis_map = {case["U_id"]: case["Case"]["Case Diagnosis"] for case in case_topics}

    # Tạo danh sách dữ liệu
    data = []
    for desc in descriptions:
        image_name = desc["image"] + ".png"
        image_path = os.path.join(images_dir, image_name)

        if os.path.exists(image_path):
            caption = desc["Description"]["Caption"]
            prompt = f"<image>\nCaption: {caption}"
            data.append({
                "image_path": image_path,
                "text": prompt,
                "label": caption
            })
        else:
            print(f"Warning: Image not found: {image_path}")

    # Tạo dataset từ danh sách
    dataset = Dataset.from_list(data)

    if len(dataset) == 0:
        raise ValueError("No valid data samples found. Check image files and Descriptions.json.")

    # Chia train/test
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # In ra các mẫu test sau khi xử lý và trước khi token hóa
    print("Các mẫu test sau khi xử lý và trước khi token hóa:")
    for i, sample in enumerate(dataset["test"]):
        print(f"Mẫu {i + 1}:")
        print(f"  Image Path: {sample['image_path']}")
        print(f"  Text (Prompt): {sample['text']}")
        print(f"  Label (Caption): {sample['label']}")
        print("-" * 50)


    # Hàm preprocess
    def preprocess_function(examples):
        try:
            images = [Image.open(img_path).convert("RGB") for img_path in examples["image_path"]]
        except Exception as e:
            raise ValueError(f"Error loading images: {e}")

        texts = examples["text"]
        labels = examples["label"]

        # Process images với image_processor và đảm bảo là tensor
        pixel_values = torch.stack([torch.from_numpy(image_processor(img).numpy()) for img in images])

        # Tokenize text (prompt)
        input_ids = tokenizer(
            texts,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        # Tokenize labels (caption)
        labels_ids = tokenizer(
            labels,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels_ids
        }


    # Áp dụng preprocess
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    # Lưu dataset
    output_dir = "./medpix_processed_local_test"
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)

    print(f"Dataset prepared with {len(dataset['train'])} train and {len(dataset['test'])} test samples.")