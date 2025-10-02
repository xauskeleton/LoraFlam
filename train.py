import torch
import torch.nn as nn
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
from Collator import collator
import os

os.environ["WANDB_MODE"] = "disabled"

# LLAVA MODEL DEFINITION
class LLaVAModel(nn.Module):
    """
    LLaVA: Visual instruction tuning
    - CLIP vision encoder
    - Linear projection
    - LLM (Vicuna/LLaMA/Mistral)
    """
    def __init__(self, vision_encoder, lang_model, image_token_id):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_model = lang_model
        self.image_token_id = image_token_id
        
        self.vision_hidden_size = vision_encoder.config.hidden_size
        self.lang_hidden_size = lang_model.config.hidden_size
        
        # Projection layer: vision -> language dimension
        self.mm_projector = nn.Linear(
            self.vision_hidden_size,
            self.lang_hidden_size
        )
        
        print(f"\n✓ LLaVA Model initialized:")
        print(f"  Vision: {self.vision_hidden_size}D -> Language: {self.lang_hidden_size}D")

    def encode_images(self, images):
        """Encode images: CLIP -> mean pool -> projection"""
        with torch.no_grad():
            image_features = self.vision_encoder(images).last_hidden_state
        
        # Mean pool patches: (B, num_patches, hidden) -> (B, hidden)
        image_features = image_features.mean(dim=1)
        
        # Project to language space
        image_features = self.mm_projector(image_features)
        
        return image_features

    def forward(self, pixel_values, input_ids, attention_mask, num_images, labels=None):
        """
        Forward pass
        
        Args:
            pixel_values: (B*N, 3, 224, 224) - all images flattened
            input_ids: (B, D) - text with <image> tokens at beginning
            attention_mask: (B, D)
            num_images: (B,) - number of images per sample
            labels: (B, D) - for training
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Convert num_images to tensor if needed
        if not isinstance(num_images, torch.Tensor):
            num_images = torch.tensor(num_images, device=device)
        
        # 1. Encode all images
        if pixel_values.size(0) > 0:
            image_features = self.encode_images(pixel_values)  # (B*N, lang_hidden)
        else:
            image_features = None
        
        # 2. Get text embeddings
        text_embeds = self.lang_model.get_input_embeddings()(input_ids)  # (B, D, lang_hidden)
        
        # 3. Replace <image> tokens at beginning with image features
        if image_features is not None:
            img_idx = 0
            
            for b in range(B):
                n_img = num_images[b].item() if isinstance(num_images[b], torch.Tensor) else num_images[b]
                
                if n_img > 0:
                    # Collator đã đặt image tokens ở đầu sequence
                    for i in range(n_img):
                        if img_idx < image_features.size(0):
                            text_embeds[b, i] = image_features[img_idx]
                            img_idx += 1
        
        # 4. Forward through language model
        outputs = self.lang_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs


# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================

def main():
    print("="*60)
    print("LLaVA TRAINING SCRIPT")
    print("="*60)
    
    # ---------------------------
    # 1. Load Dataset
    # ---------------------------
    print("\n[1/5] Loading dataset...")
    dataset = load_from_disk("MedPix-2-0/dataset_hf")
    
    if "train" not in dataset:
        dataset = dataset.train_test_split(test_size=0.05)
    
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # ---------------------------
    # 2. Load Models
    # ---------------------------
    print("\n[2/5] Loading models...")
    
    # Vision encoder - CLIP ViT-Large
    print("  Loading vision encoder...")
    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Language model - Chọn 1 trong các options:
    print("  Loading language model...")
    
    # OPTION 1: Vicuna (khuyên dùng nếu có access)
    # lang_model_name = "lmsys/vicuna-7b-v1.5"
    
    # OPTION 2: Mistral (open, không cần request access)
    lang_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # OPTION 3: OPT (nhỏ, test nhanh)
    # lang_model_name = "facebook/opt-1.3b"
    
    # OPTION 4: LLaMA 2 (cần request access)
    # lang_model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    lang_model = AutoModelForCausalLM.from_pretrained(
        lang_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add image token
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    
    # Resize embeddings
    lang_model.resize_token_embeddings(len(tokenizer))
    
    print(f"  ✓ Vocab size: {len(tokenizer)}")
    print(f"  ✓ Image token ID: {image_token_id}")
    print(f"  ✓ Vision hidden: {vision_encoder.config.hidden_size}")
    print(f"  ✓ Language hidden: {lang_model.config.hidden_size}")
    
    # ---------------------------
    # 3. Build LLaVA Model
    # ---------------------------
    print("\n[3/5] Building LLaVA model...")
    
    model = LLaVAModel(vision_encoder, lang_model, image_token_id)
    
    # Freeze vision encoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    
    # Apply LoRA to language model
    print("  Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # For LLaMA/Mistral
        task_type="CAUSAL_LM",
    )
    model.lang_model = get_peft_model(model.lang_model, lora_config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ---------------------------
    # 4. Setup Training
    # ---------------------------
    print("\n[4/5] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir="./llava_lora_out",
        per_device_train_batch_size=4,      # Adjust based on GPU memory
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=False,                       # Use bf16 for better stability
        remove_unused_columns=False,
        gradient_accumulation_steps=4,       # Effective batch = 4 * 4 = 16
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=lambda samples: collator(samples, image_token_id=image_token_id),
    )
    
    # ---------------------------
    # 5. Train!
    # ---------------------------
    print("\n[5/5] Starting training...")
    print("="*60)
    
    trainer.train()
    
    # ---------------------------
    # 6. Save
    # ---------------------------
    print("\n" + "="*60)
    print("Saving model...")
    
    trainer.save_model("./llava_lora_final")
    model.lang_model.save_pretrained("./llava_lora_final/language_model")
    torch.save(model.mm_projector.state_dict(), "./llava_lora_final/mm_projector.pt")
    
    print("✓ Training complete!")
    print(f"  Model saved to: ./llava_lora_final/")
    print("="*60)


if __name__ == "__main__":
    main()