import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPImageProcessor,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model
from Collator import collator  # collator bạn đã viết sẵn
import os
os.environ["WANDB_MODE"] = "disabled"

# ---------------------------
# 1. Dataset + DataLoader
# ---------------------------
print("Loading dataset...")
dataset = load_from_disk("dataset_hf")

# train/val split nếu chưa có
if "train" not in dataset:
    dataset = dataset.train_test_split(test_size=0.05)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

# ---------------------------
# 2. Load Vision + Language Model
# ---------------------------
print("Loading vision encoder and language model...")

vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

lang_model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
tokenizer.pad_token = tokenizer.eos_token  # phòng khi thiếu pad token
lang_model = AutoModelForCausalLM.from_pretrained(lang_model_name)

# ---------------------------
# 3. Define Flamingo-style wrapper
# ---------------------------
class FlamingoMini(nn.Module):
    def __init__(self, vision_encoder, lang_model, vision_hidden=512, cross_attn_every_n=4):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_model = lang_model

        self.vision_proj = nn.Linear(vision_encoder.config.hidden_size, vision_hidden)

        self.cross_attn_every_n = cross_attn_every_n
        self.cross_attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=vision_hidden, num_heads=8, batch_first=True)
            for _ in range(len(lang_model.model.decoder.layers) // cross_attn_every_n)
        ])

        # map hidden của LM về vision_hidden nếu khác dim
        self.text_proj = nn.Linear(lang_model.config.hidden_size, vision_hidden)
        self.text_unproj = nn.Linear(vision_hidden, lang_model.config.hidden_size)

    def forward(self, pixel_values, input_ids, attention_mask, num_images, labels=None):
        B = input_ids.size(0)

        # ---- 1. Vision encoder ----
        vision_out = self.vision_encoder(pixel_values=pixel_values)
        vision_embeds = self.vision_proj(vision_out.last_hidden_state)  # (B*T, N, 512)

        # gộp lại cho từng sample
        N = vision_embeds.size(1)
        T = num_images.max().item()
        vision_embeds = vision_embeds.view(B, T*N, -1)  # (B, T*N, 512)

        # ---- 2. Language model hidden ----
        outputs = self.lang_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]  # (B, L, 1024)

        # ---- 3. Projection sang vision space ----
        hidden_proj = self.text_proj(hidden)  # (B, L, 512)

        # ---- 4. Cross attention ----
        for i, block in enumerate(self.cross_attn_blocks):
            if i * self.cross_attn_every_n < hidden_proj.size(1):
                attn_out, _ = block(hidden_proj, vision_embeds, vision_embeds)
                hidden_proj = hidden_proj + attn_out

        # ---- 5. Chiếu ngược về dim của LM ----
        hidden_final = self.text_unproj(hidden_proj)  # (B, L, 1024)

        # ---- 6. Tính logits ----
        logits = self.lang_model.lm_head(hidden_final)  # (B, L, V)

        # ---- 7. Loss ----
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


# ---------------------------
# 4. Wrap model + add LoRA
# ---------------------------
print("Building FlamingoMini with LoRA...")

model = FlamingoMini(vision_encoder, lang_model)

# gắn LoRA cho LM
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "proj", "lm_head"],
    task_type="CAUSAL_LM",
)
model.lang_model = get_peft_model(model.lang_model, config)

# ---------------------------
# 5. Training
# ---------------------------
training_args = TrainingArguments(
    output_dir="./flamingo_lora_out",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    num_train_epochs=5,
    logging_steps=20,
    save_strategy="epoch",
    fp16=True,
    remove_unused_columns=False,  # cần để collator không bị cắt field
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=lambda samples: collator(samples),  # dùng collator.py
)

print("Start training...")
trainer.train()
