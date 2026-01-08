import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading datasets...")
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.json",
        "validation": "data/val.json"
    }
)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Loading processor...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Tamil",
    task="transcribe"
)

def preprocess(batch):
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        text=batch["text"]
    )

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = inputs.labels
    return batch

dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names,
    num_proc=2
)

print("Loading Whisper-small model...")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
)

# Force Tamil decoding
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="tamil",
    task="transcribe"
)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.to(DEVICE)

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = torch.stack(
            [torch.tensor(f["input_features"]) for f in features]
        )

        labels = [torch.tensor(f["labels"]) for f in features]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_features": input_features,
            "labels": labels
        }

training_args = TrainingArguments(
    output_dir="outputs/lora-whisper-small",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorSpeechSeq2Seq(processor)
)

print("Starting training...")
trainer.train()

model.save_pretrained("outputs/lora-whisper-small")
processor.save_pretrained("outputs/lora-whisper-small")

print("LoRA fine-tuning complete")
