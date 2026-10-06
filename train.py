import os
import shutil
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset import SFTDataset
from torch.utils.data import ConcatDataset

# 强制关闭 Tokenizer 并行，防止 DDP 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# A40 算力优化
torch.backends.cuda.matmul.allow_tf32 = True

CONFIG = {
    "model_path": {
        "llama-3.2-3b-instruct": "./llama-3b-model",
        "qwen3-4b": "./qwen-4b-model",
        "gemma-3-4b-it": "./gemma-4b-model",
    },
    # "model": "llama-3.2-3b-instruct",
    # "model": "qwen3-4b",
    "model": "gemma-3-4b-it",
    "data_path": [
        # "./Dataset/train/ibmuav_augment.csv",
        # "./Dataset/train/opencoss_augment.csv",
        # "./Dataset/train/pure_augment.csv",
        # "./Dataset/train/uav_augment.csv",
        # "./Dataset/train/worldvista_augment.csv",
        # "./Dataset/train/worldvista.csv",
        # "./Dataset/train/alice.csv",
        # "./Dataset/train/cn.csv",
        "./Dataset/train.csv",
        # "./Dataset/rare_train.csv",
    ],
    "logits_path": "./data/teacher_logits",
    # "output_dir": "./output/llama_distill",
    "output_dir": "./output/qwen_distill",
    "batch_size": 2,  # A40 充裕显存
    "gradient_accum": 8,  # 1 步一同步，效率最高
    "learning_rate": 1e-5,
    "epochs": 4,
    "max_seq_length": 512,
    "distill_weight": 0,  # 0 为纯 SFT
    "temperature": 2.0,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "loss_stop_threshold": 0.1,
}

weight_suffix = str(CONFIG["distill_weight"]).replace(".", "-")
CONFIG["output_dir"] = f"{CONFIG['output_dir']}_{CONFIG['model']}_{weight_suffix}"
FINAL_SAVE_PATH = os.path.join(CONFIG["output_dir"], "final_merged_model")

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class LossStoppingCallback(TrainerCallback):
    """当训练 Loss 低于设定阈值时终止训练的回调函数"""
    def __init__(self, stop_threshold: float):
        self.stop_threshold = stop_threshold

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        # 检查日志中是否有 'loss' 字段
        if logs is not None and "loss" in logs:
            current_loss = logs["loss"]
            if current_loss <= self.stop_threshold:
                print(f"\n[终止提醒] 当前 Loss ({current_loss:.6f}) 已低于阈值 ({self.stop_threshold})，正在终止训练...")
                control.should_training_stop = True

def clean_dirs():
    if os.path.exists(CONFIG["output_dir"]):
        shutil.rmtree(CONFIG["output_dir"])


def compute_kl_loss(student_logits, teacher_probs, teacher_indices, labels, temp=2.0):
    log_p_student_all = F.log_softmax(student_logits / temp, dim=-1)
    log_p_student_topk = torch.gather(log_p_student_all, -1, teacher_indices)

    kl_div = teacher_probs * (torch.log(teacher_probs + 1e-6) - log_p_student_topk)
    mask = (labels != -100).unsqueeze(-1)
    kl_loss = (kl_div * mask).sum() / (mask.sum() + 1e-6)
    return kl_loss * (temp**2)


class UniversalTrainer(Trainer):
    def __init__(self, distill_weight=0.5, temp=2.0, **kwargs):
        super().__init__(**kwargs)
        self.distill_weight = distill_weight
        self.temp = temp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        t_probs = inputs.pop("teacher_probs", None)
        t_indices = inputs.pop("teacher_indices", None)

        outputs = model(**inputs)
        student_logits = outputs.logits
        ce_loss = outputs.loss

        if t_probs is not None and self.distill_weight > 0:
            kl_loss = compute_kl_loss(
                student_logits, t_probs, t_indices, inputs["labels"], temp=self.temp
            )
            loss = (1 - self.distill_weight) * ce_loss + self.distill_weight * kl_loss
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


def run_training():
    # --- 分布式环境识别 ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    model_key = CONFIG["model"]
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"][model_key])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 模型加载 (严禁 device_map="auto") ---
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"][model_key],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=CONFIG["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # dataset = SFTDataset(
    #     data_path=CONFIG["data_path"], logits_path=CONFIG["logits_path"],
    #     tokenizer=tokenizer, max_seq_len=CONFIG["max_seq_length"],
    #     rank=max(0, local_rank), world_size=world_size
    # )

    all_sub_datasets = []
    for d_path, l_path in zip(CONFIG["data_path"], CONFIG["logits_path"]):
        sub_dataset = SFTDataset(
            data_path=d_path,
            logits_path=l_path,
            tokenizer=tokenizer,
            max_seq_len=CONFIG["max_seq_length"],
            rank=max(0, local_rank),
            world_size=world_size,
        )
        all_sub_datasets.append(sub_dataset)
    dataset = ConcatDataset(all_sub_datasets)

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accum"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["epochs"],
        bf16=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
    )

    trainer = UniversalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
        distill_weight=CONFIG["distill_weight"],
        temp=CONFIG["temperature"],
        callbacks=[LossStoppingCallback(stop_threshold=CONFIG["loss_stop_threshold"])]
    )

    trainer.train()

    # --- 仅在主进程保存结果 ---
    if local_rank <= 0:
        print("--- 训练结束，正在保存最终模型 ---")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(FINAL_SAVE_PATH)
        tokenizer.save_pretrained(FINAL_SAVE_PATH)


if __name__ == "__main__":
    l_rank = int(os.environ.get("LOCAL_RANK", 0))
    if l_rank == 0:
        clean_dirs()
    run_training()
