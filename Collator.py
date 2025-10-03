import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import BlipProcessor

# Load dataset và tokenizer
ds = load_from_disk("MedPix-2-0/dataset_hf")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

# Collator Flamingo-style
def collator(samples, image_token_id=image_token_id):
    batch_size = len(samples)

    # --- 1. Pad pixel_values ---
    n_imgs_list = [s["pixel_values"].shape[0] for s in samples]
    max_n_img = max(n_imgs_list)

    batch_pixel_values = []
    for s in samples:
        n_img = s["pixel_values"].shape[0]
        pixel_vals = s["pixel_values"]
        
        # Đảm bảo pixel_values có đúng shape (N, C, H, W)
        if pixel_vals.dim() == 3:
            # Nếu chỉ có 1 ảnh và shape là (C, H, W), thêm batch dim
            pixel_vals = pixel_vals.unsqueeze(0)
        
        if n_img < max_n_img:
            # QUAN TRỌNG: Pad với shape khớp với pixel_vals
            C, H, W = pixel_vals.shape[1], pixel_vals.shape[2], pixel_vals.shape[3]
            pad = torch.zeros((max_n_img - n_img, C, H, W), dtype=pixel_vals.dtype)
            pixel_values = torch.cat([pixel_vals, pad], dim=0)
        else:
            pixel_values = pixel_vals
        batch_pixel_values.append(pixel_values)
    
    batch_pixel_values = torch.stack(batch_pixel_values, dim=0)    # (B, T, C, H, W)

    B, T, C, H, W = batch_pixel_values.shape
    # Flatten thành (B*T, C, H, W) để đưa vào vision encoder
    batch_pixel_values = batch_pixel_values.view(B*T, C, H, W)

    # --- 2. Pad text + prepend image tokens ---
    seq_lens = [s["input_ids"].shape[0] + max_n_img for s in samples]
    max_seq_len = max(seq_lens)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for s in samples:
        n_img = s["pixel_values"].shape[0] if s["pixel_values"].dim() == 4 else 1

        # image tokens thật
        image_tokens = torch.full((n_img,), image_token_id, dtype=torch.long)
        # image tokens padding
        image_tokens_pad = torch.full((max_n_img - n_img,), 0, dtype=torch.long)

        # concat image tokens (cả thật + pad) + text
        input_ids = torch.cat([image_tokens, image_tokens_pad, s["input_ids"]], dim=0)

        # attention mask: 1 cho ảnh thật, 0 cho ảnh pad
        image_mask = torch.cat([
            torch.ones(n_img, dtype=torch.long),
            torch.zeros(max_n_img - n_img, dtype=torch.long)
        ], dim=0)
        attention_mask = torch.cat([image_mask, s["text_attention_mask"]], dim=0)

        # chỉ pad input_ids và attention_mask
        pad_len = max_seq_len - input_ids.shape[0]
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        # labels giữ nguyên từ dataset, chỉ cần pad chiều
        labels = s["labels"]
        if labels.shape[0] < max_seq_len:
            pad_len = max_seq_len - labels.shape[0]
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "pixel_values": batch_pixel_values,
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
        "num_images": torch.tensor(n_imgs_list)
    }