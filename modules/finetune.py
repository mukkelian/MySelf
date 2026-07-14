"""
Trains your chosen model on your Q&A pairs (and any plain text files you
added), so it learns to answer the way you taught it. Uses a lightweight
training method (LoRA) by default so it can run on a normal laptop.
"""

import os
import shutil
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from . import model_manager

PROMPT_TEMPLATE = "Question: {question}\nAnswer: {answer}"


def _build_dataset(pairs, text_chunks, tokenizer, max_length):
    if getattr(tokenizer, "chat_template", None):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["question"]}, {"role": "assistant", "content": p["answer"]}],
                tokenize=False,
            )
            for p in pairs
        ]
    else:
        texts = [PROMPT_TEMPLATE.format(**p) + tokenizer.eos_token for p in pairs]

    texts += [chunk + tokenizer.eos_token for chunk in text_chunks]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": texts})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def train(model_path: str, qa_pairs: list, text_chunks: list, params: dict, log=print):
    """Train the model and save the result to params['staging_dir'].

    If full_finetune is on, the whole model is retrained (slower, more
    thorough). Otherwise only small LoRA adapters are trained, which is
    much faster, then blended back into the model before saving.
    """
    if not qa_pairs and not text_chunks:
        raise ValueError("No Q&A pairs or text found in the selected dataset folder.")

    full_finetune = bool(params.get("full_finetune"))

    log(f"Loading base model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = model_manager.get_device(params.get("device", "auto"))
    log(f"Training on device: {device}")
    base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    if full_finetune:
        log("Full fine-tuning: training all model parameters ...")
        model = base_model
    else:
        log("Wrapping model with LoRA adapters ...")
        lora_config = LoraConfig(
            r=params.get("lora_r", 8),
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.05),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    log(f"Tokenizing {len(qa_pairs)} Q&A pairs and {len(text_chunks)} text chunks ...")
    dataset = _build_dataset(qa_pairs, text_chunks, tokenizer, params.get("max_length", 256))

    staging_dir = params.get("staging_dir", "models/_staging_finetune")
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=staging_dir,
        num_train_epochs=params.get("epochs", 3),
        per_device_train_batch_size=params.get("batch_size", 2),
        learning_rate=params.get("learning_rate", 2e-4),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    log("Training started ...")
    trainer.train()
    log("Training finished. Saving model ...")

    if full_finetune:
        final_model = model
    else:
        final_model = model.merge_and_unload()

    final_model.save_pretrained(staging_dir)
    tokenizer.save_pretrained(staging_dir)

    model_manager.clear_cache()
    log(f"Fine-tuned model staged at {staging_dir}")
    return staging_dir
