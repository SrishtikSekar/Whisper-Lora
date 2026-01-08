import torch
import re
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

wer_metric = evaluate.load("wer")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to(device)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/lora-whisper-small"
).to(device)

# Force Tamil decoding
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ta",
    task="transcribe"
)
model.config.forced_decoder_ids = forced_decoder_ids

dataset = load_dataset(
    "fsicoli/common_voice_19_0",
    "ta",
    split="test[:150]"
)

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\u0B80-\u0BFF\s]", "", text)  # keep Tamil chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

preds, refs = [], []

for ex in dataset:
    audio = ex["audio"]["array"]

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"].to(device),
            max_new_tokens=64,
            num_beams=1,
            do_sample=False
        )

    pred = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    preds.append(normalize(pred))
    refs.append(normalize(ex["sentence"]))

wer = wer_metric.compute(predictions=preds, references=refs)
print(f"Fine-tuned WER: {wer * 100:.2f}%")
