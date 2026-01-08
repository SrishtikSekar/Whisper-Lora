import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading test dataset...")
dataset = load_dataset(
    "json",
    data_files="data/test.json",
    split="train"
)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Loading Whisper-small (base)...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Tamil",
    task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to(DEVICE)

# Force Tamil decoding
forced_ids = processor.get_decoder_prompt_ids(
    language="tamil",
    task="transcribe"
)
model.config.forced_decoder_ids = forced_ids

preds, refs = [], []

print("Evaluating base model...")
for sample in tqdm(dataset):
    inputs = processor(
        sample["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_features.to(DEVICE),
            max_new_tokens=128
        )

    transcription = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    preds.append(transcription)
    refs.append(sample["text"])

base_wer = wer(refs, preds)

with open("baselines/base_wer.txt", "w") as f:
    f.write(f"Base Whisper-small WER: {base_wer * 100:.2f}%\n")

print(f"\n✅ Base WER: {base_wer * 100:.2f}%")
