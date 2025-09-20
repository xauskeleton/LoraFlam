# Import các thư viện cần thiết
import torch
import warnings
import logging
import os
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
from open_flamingo import create_model_and_transforms
import torch.distributed as dist
import torch.multiprocessing as mp

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bỏ qua cảnh báo PyTorch
warnings.filterwarnings("ignore", category=UserWarning)

# Thiết lập multi-GPU
def setup_distributed(rank, world_size):
    """Khởi tạo môi trường multi-GPU."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Dọn dẹp tài nguyên multi-GPU
def cleanup_distributed():
    """Hủy môi trường multi-GPU."""
    dist.destroy_process_group()

# Định nghĩa DataCollator
class MultimodalDataCollator:
    def __call__(self, features):
        try:
            pixel_values = torch.stack([torch.tensor(f["pixel_values"], dtype=torch.bfloat16) for f in features])
            input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
            labels = torch.stack([torch.tensor(f["labels"]) for f in features])
            return {
                "vision_x": pixel_values.unsqueeze(1).unsqueeze(1),
                "lang_x": input_ids,
                "labels": labels
            }
        except Exception as e:
            logger.error(f"Rank {rank}: Lỗi trong DataCollator: {e}")
            raise

# Định nghĩa hàm tính toán metrics
def compute_metrics(eval_pred):
    """Tính toán metrics cho mô hình."""
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}

# Cấu hình LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["Wqkv", "out_proj", "up_proj", "down_proj"]
)

# Cấu hình tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./openflamingo-lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    bf16=True,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    do_eval=True,
    eval_steps=500,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    report_to="none"
)

# Hàm huấn luyện mô hình
def train_model(rank, world_size):
    """Hàm huấn luyện mô hình với multi-GPU."""
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    logger.info(f"Process {rank} sử dụng device: {device}")

    # Tải mô hình và tokenizer
    try:
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Rank {rank}: Đã đặt pad_token bằng eos_token.")
    except Exception as e:
        logger.error(f"Rank {rank}: Lỗi tải mô hình: {e}")
        cleanup_distributed()
        raise

    # Chuyển mô hình sang device và dtype
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Áp dụng LoRA
    try:
        lang_encoder = model.module.lang_encoder
    except AttributeError:
        logger.error(f"Rank {rank}: Không tìm thấy lang_encoder. Kiểm tra cấu trúc mô hình:")
        for name, module in model.named_modules():
            logger.error(name)
        cleanup_distributed()
        raise

    lang_encoder = get_peft_model(lang_encoder, lora_config)
    lang_encoder.print_trainable_parameters()
    model.module.lang_encoder = lang_encoder

    # Tải dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medpix_processed_local")  # Sửa _file_ thành __file__
    if not os.path.exists(dataset_path):
        logger.error(f"Rank {rank}: Dataset không tồn tại tại {dataset_path}")
        cleanup_distributed()
        raise FileNotFoundError(f"Dataset không tìm thấy.")

    try:
        dataset = load_from_disk(dataset_path)
        logger.info(f"Rank {rank}: Đã tải dataset với {len(dataset['train'])} train và {len(dataset['test'])} test.")
    except Exception as e:
        logger.error(f"Rank {rank}: Lỗi tải dataset: {e}")
        cleanup_distributed()
        raise

    data_collator = MultimodalDataCollator()

    # Khởi tạo Trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    except Exception as e:
        logger.error(f"Rank {rank}: Lỗi khởi tạo Trainer: {e}")
        cleanup_distributed()
        raise

    # Bắt đầu fine-tune
    logger.info(f"Rank {rank}: Bắt đầu quá trình fine-tune...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Rank {rank}: Lỗi trong quá trình train: {e}")
        cleanup_distributed()
        raise

    # Lưu LoRA adapter (chỉ rank 0)
    if rank == 0:
        os.makedirs("./lora-adapter", exist_ok=True)
        try:
            lang_encoder.save_pretrained("./lora-adapter")
            logger.info("Đã lưu LoRA adapter vào ./lora-adapter")
        except Exception as e:
            logger.error(f"Rank {rank}: Lỗi lưu LoRA adapter: {e}")
            cleanup_distributed()
            raise

    cleanup_distributed()

# Chạy huấn luyện
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.error(f"Chỉ phát hiện {world_size} GPU. Yêu cầu ít nhất 2 GPU.")
        raise RuntimeError("Cần nhiều GPU để huấn luyện.")
    mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)