# all_ft_runpod.py
from dotenv import load_dotenv
load_dotenv()  # .env 구문 자동 로드

import os
import shutil
import torch
import configparser
from huggingface_hub import HfApi, login
from datasets import load_dataset, Features, Value
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig, TrainerCallback
)


class LossLoggerCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            with open(self.save_path, "a") as f:
                f.write(f"Step {step}: loss = {loss:.4f}\n")


class FullFTTrainer:
    def __init__(self, model_id, file_path, log_path, output_dir, repo_id, epoch, dtype='fp32'):
        self.model_id = model_id
        self.file_path = file_path
        self.log_path = log_path
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.epoch = epoch
        self.dtype = dtype
        self.api = HfApi()

    def initialize_environment(self):
        login(token=os.environ.get("HF_TOKEN"))
        print("✅ Hugging Face 로그인 완료:", self.api.whoami()["name"])

    def load_and_prepare_dataset(self):
        temp_cache_dir = "./tmp_cache"
        os.makedirs(temp_cache_dir, exist_ok=True)
        features = Features({
            "content": Value("string"),
            "emotion": Value("string"),
            "post_type": Value("string"),
            "transformed_content": Value("string")
        })
        self.dataset = load_dataset(
            "json",
            data_files=self.file_path,
            features=features,
            split="train",
            cache_dir=temp_cache_dir,
            verification_mode="no_checks"
        )
        shutil.rmtree(temp_cache_dir, ignore_errors=True)

    def format_and_tokenize(self, example):
        text = (
            f"### content:\n{example['content']}\n"
            f"### emotion:\n{example['emotion']}\n"
            f"### post_type:\n{example['post_type']}\n\n"
            f"### transformed_content:\n{example['transformed_content']}"
        )
        tokenized = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def tokenize_dataset(self):
        self.tokenized_dataset = self.dataset.map(self.format_and_tokenize, remove_columns=self.dataset.column_names)

    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=self.epoch,
            fp16=(self.dtype == 'fp16'),
            bf16=(self.dtype == 'bf16'),
            logging_steps=10,
            save_strategy="no",
            logging_strategy="steps",
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=False
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            callbacks=[LossLoggerCallback(save_path=self.log_path)],
            data_collator=data_collator
        )
        return trainer

    def train_and_upload(self, trainer):
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True)
        self.api.upload_folder(folder_path=self.output_dir, repo_id=self.repo_id, repo_type="model")
        print(f"✅ 모델 업로드 완료: https://huggingface.co/{self.repo_id}")
        shutil.rmtree(self.output_dir, ignore_errors=True)


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")

    section = "haebo/Meow-kanana1.5-2.1B_FullFT_fp32_0527"
    selected_model = config[section]['selected_model']
    selected_name = config[section]['selected_name']
    selected_method = config[section]['selected_method']
    selected_dataset = config[section]['selected_dataset']
    selected_code = config[section]['selected_code']
    selected_dtype = config[section]['selected_dtype']
    selected_epoch = int(config[section]['selected_epoch'])

    file_path = f"dataset/{selected_dataset}"
    log_path = f"log/{selected_name}-{selected_method}-{selected_dtype}-{selected_code}-log.txt"
    output_dir = "finetuned-ktb"
    repo_id = f"haebo/Meow-{selected_name}_{selected_method}_{selected_dtype}_{selected_code}"

    os.makedirs("log", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    trainer = FullFTTrainer(selected_model, file_path, log_path, output_dir, repo_id, selected_epoch, selected_dtype)
    trainer.initialize_environment()
    trainer.load_and_prepare_dataset()
    trainer.setup_model_and_tokenizer()
    trainer.tokenize_dataset()
    trainer_instance = trainer.setup_trainer()
    trainer.train_and_upload(trainer_instance)


if __name__ == "__main__":
    main()
