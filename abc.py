import json
import csv
from pathlib import Path
import random
import re
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import Dataset

# --- config ---
DESCRIPTIONS = "Descriptions.json"
CASE_TOPIC = "Case_topic.json"
IMAGES_OVERVIEW = "images_overview.csv"
IMAGES_DIR = "images"
MODEL_NAME = "Salesforce/blip-vqa-base"  # thay BERT bang BLIP
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
MAX_LEN = 256


# ----------------

def clean_text(s):
    if not s: return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


# === load data ===
with open(DESCRIPTIONS, "r", encoding="utf-8") as f:
    desc_list = json.load(f)
img2desc = {d["image"]: d for d in desc_list if "image" in d}

with open(CASE_TOPIC, "r", encoding="utf-8") as f:
    cases = json.load(f)
uid2case = {c["U_id"]: c for c in cases if "U_id" in c}

img_overview = {}
if Path(IMAGES_OVERVIEW).exists():
    with open(IMAGES_OVERVIEW, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            key = r.get("ID_Image")
            if key:
                img_overview[key] = r

# === models ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = BlipProcessor.from_pretrained(MODEL_NAME).tokenizer


# === tokenize ===
def tokenize_pair(input_text, output_text):
    enc_in = tokenizer(
        input_text, max_length=MAX_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    enc_out = tokenizer(
        output_text, max_length=MAX_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    enc_out["input_ids"][enc_out["input_ids"] == tokenizer.pad_token_id] = -100
    return (
        enc_in["input_ids"].squeeze(0),
        enc_in["attention_mask"].squeeze(0),
        enc_out["input_ids"].squeeze(0),
    )


# === group images by case ===
uid2images = defaultdict(list)
for image_id, desc in img2desc.items():
    uid2images[desc["U_id"]].append(image_id)

# === build dataset (n + m) ===
records = []

for uid, image_ids in uid2images.items():
    if uid not in uid2case:
        continue
    case = uid2case[uid]

    # -------- CASE sample (m) --------
    all_tensors = []
    for image_id in image_ids:
        image_path = img_overview.get(image_id, {}).get("image_path")
        if not image_path:
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".png"]:
                candidate = Path(IMAGES_DIR) / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = str(candidate);
                    break
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            img_tensor = inputs["pixel_values"].squeeze(0)  # (3,224,224)
            all_tensors.append(img_tensor)

    if all_tensors:
        pixel_values = torch.stack(all_tensors, dim=0)  # (N,3,224,224)
        diagnosis = clean_text(case.get("Case", {}).get("Case Diagnosis") or "")
        exam = clean_text(case.get("Case", {}).get("Exam") or "")
        findings = clean_text(case.get("Case", {}).get("Findings") or "")
        history = clean_text(case.get("Case", {}).get("History") or "")
        topic_disc = clean_text(case.get("Topic", {}).get("Disease Discussion") or "")
        topic_summary = ". ".join(topic_disc.split(".")[:2]) if topic_disc else ""

        input_text = f"History: {history}\nExam: {exam}\nFindins: {findings}"
        output_text = f"Diagnosis: {diagnosis}\nDiscussion: {topic_summary}"

        input_ids, attn_mask, labels = tokenize_pair(input_text, output_text)

        records.append({
            "type": "case",
            "U_id": uid,
            "pixel_values": pixel_values,  # (N,3,224,224)
            "input_ids": input_ids,
            "text_attention_mask": attn_mask,
            "labels": labels
        })

    # -------- SINGLE samples (n) --------
    for image_id in image_ids:
        desc = img2desc[image_id]
        d = desc["Description"]
        caption = clean_text(d.get("Caption") or "")
        modality = clean_text(d.get("Modality") or "")
        location = clean_text(desc.get("Location") or "")
        age, sex = clean_text(d.get("Age") or ""), clean_text(d.get("Sex") or "")
        short_info = f"Age: {age}, Sex: {sex}".strip(", ")

        input_text = f"Modality: {modality}\nLocation: {location}\n{short_info}"  # Chỉnh caption từ bên input text sang output
        output_text = f"Caption: {caption}"

        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = Path(IMAGES_DIR) / f"{image_id}{ext}"
            if candidate.exists(): image_path = str(candidate); break

        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            img_tensor = inputs["pixel_values"]  # xóa .squeeze(0) để thành tensor 4d

            input_ids, attn_mask, labels = tokenize_pair(input_text, output_text)

            records.append({
                "type": "single",
                "U_id": uid,
                "image_id": image_id,
                "pixel_values": img_tensor,  # (1,3,224,224)
                "input_ids": input_ids,
                "text_attention_mask": attn_mask,
                "labels": labels
            })

# === split train/val/test ===
uids = list(uid2images.keys())
random.Random(SPLIT_SEED).shuffle(uids)
n = len(uids);
n_train = int(n * TRAIN_RATIO);
n_val = int(n * VAL_RATIO)
train_uids = set(uids[:n_train]);
val_uids = set(uids[n_train:n_train + n_val])
test_uids = set(uids[n_train + n_val:])


def which_split(uid): return "train" if uid in train_uids else ("val" if uid in val_uids else "test")


for r in records:
    r["split"] = which_split(r["U_id"])

print(f"✅ Dataset built: {len(records)} samples (single + case)")

from datasets import Dataset, concatenate_datasets

# Chia nhỏ thành các batch để tránh lỗi bộ nhớ
batch_size = 1000  # Điều chỉnh theo khả năng RAM của bạn
datasets = []

print(f"Chia {len(records)} records thành các batch {batch_size}...")

for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    print(f"Xử lý batch {i // batch_size + 1}: records {i} đến {i + len(batch)}")
    ds_batch = Dataset.from_list(batch)
    datasets.append(ds_batch)

# Ghép lại thành dataset hoàn chỉnh
ds = concatenate_datasets(datasets)
print("✅ Đã tạo dataset thành công từ các batch")

# Tiếp tục các bước sau
ds.set_format(
    type="torch",
    columns=["input_ids", "text_attention_mask", "labels", "pixel_values"]
)

ds.save_to_disk("dataset_hf")
print("✅ Dataset saved to 'dataset_hf' directory")





