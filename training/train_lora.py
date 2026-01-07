from datasets import load_dataset,Audio
from transformer import(
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig,get_peft_model

model_name="openai/whisper-small"

processor=WhisperProcessor.from_pretrained(
    model_name,
    language="Tamil",
    task="transcribe"
)

model=WhisperforConditionGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids=None
model.config.suppress_tokens=[]

lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model=get_peft_model(model,lora_config)
model.print_trainable_parameters()

dataset=load_dataset("json",data_files={
    "train":"data/train.json",
    "validation":"data/val.json"
})

dataset=dataset.cast_column("audio",Audio(sampling_rate=160000))

def preprocess(batch):
    audio=batch["audio"]
    inputs=processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    )
    with processor.as_target_processor():
        labels=processor(batch["text"]).input_ids

    batch["input_features"]=inputs.input_features[0]
    batch["labels"]=labels
    return batch

dataset=dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

args = Seq2SeqTrainingArguments(
    output_dir="outputs/lora-whisper-small",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    fp16=True,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor
)

trainer.train()