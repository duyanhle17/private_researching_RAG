import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments


# =====================
# CONFIG
# =====================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "sat_data/sat_lp_train.jsonl"
OUTPUT_DIR = "sat_lora_model"

MAX_SEQ_LEN = 512
EPOCHS = 2
LR = 2e-4

# =====================
# DEVICE (MPS / CPU)
# =====================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("🖥️ Using device:", device)

# =====================
# LOAD MODEL & TOKENIZER
# =====================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=512,
    truncation=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
)

model.to(device)

# =====================
# LORA CONFIG
# =====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =====================
# LOAD DATASET
# =====================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# =====================
# TRAINER
# =====================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=20,
    save_strategy="epoch",
    report_to=[]
)

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=dict(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         num_train_epochs=EPOCHS,
#         learning_rate=LR,
#         logging_steps=20,
#         output_dir=OUTPUT_DIR,
#         save_strategy="epoch",
#         report_to="none"
#     )
# )

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)


# =====================
# TRAIN
# =====================
trainer.train()

# =====================
# SAVE MODEL
# =====================
trainer.save_model(OUTPUT_DIR)
print("🎉 SAT LoRA model saved to:", OUTPUT_DIR)



# import torch
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer

# # =====================
# # CONFIG
# # =====================
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# DATA_PATH = "sat_data/sat_lp_train.jsonl"
# OUTPUT_DIR = "sat_lora_model"

# MAX_SEQ_LEN = 384        # ❗ GIẢM để tránh overflow
# EPOCHS = 2
# LR = 5e-5               # ❗ GIẢM LR cho dataset lớn

# # =====================
# # DEVICE (MPS / CPU)
# # =====================
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print("🖥️ Using device:", device)

# # =====================
# # LOAD TOKENIZER
# # =====================
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     model_max_length=MAX_SEQ_LEN,
#     truncation=True,
#     padding="max_length"
# )

# # =====================
# # LOAD MODEL (❗ FP32 BẮT BUỘC TRÊN MPS)
# # =====================
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float32   # ❗ KHÔNG fp16
# )

# model.to(device)

# # =====================
# # LORA CONFIG (GIỮ NGUYÊN)
# # =====================
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     task_type="CAUSAL_LM"
# )

# model = get_peft_model(model, lora_config)

# # =====================
# # LOAD DATASET
# # =====================
# dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# # =====================
# # TRAINING ARGS (ỔN ĐỊNH)
# # =====================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     num_train_epochs=EPOCHS,
#     learning_rate=LR,
#     max_grad_norm=1.0,          # ❗ CLIP GRADIENT
#     logging_steps=20,
#     save_strategy="epoch",
#     fp16=False,                 # ❗ TẮT FP16
#     bf16=False,
#     report_to=[]
# )

# # =====================
# # TRAINER
# # =====================
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=training_args
# )

# # =====================
# # TRAIN
# # =====================
# trainer.train()

# # =====================
# # SAVE MODEL
# # =====================
# trainer.save_model(OUTPUT_DIR)
# print("🎉 SAT LoRA model saved to:", OUTPUT_DIR)

