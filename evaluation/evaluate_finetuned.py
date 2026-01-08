import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import evaluate

wer_metric = evaluate.load("wer")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to("cuda")

model = PeftModel.from_pretrained(
    base_model,
    "outputs/lora-whisper-small"
).to("cuda")

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "ta",
    split="test[:150]"
)

def evaluate_model():
    preds, refs = [], []

    for ex in dataset:
        audio = ex["audio"]["array"]
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            predicted_ids = model.generate(
                **inputs,
                language="ta",
                task="transcribe"
            )

        pred = processor.decode(predicted_ids[0], skip_special_tokens=True)
        preds.append(pred.lower())
        refs.append(ex["sentence"].lower())

    return wer_metric.compute(predictions=preds, references=refs)

wer = evaluate_model()
print(f"Fine-tuned WER: {wer * 100:.2f}%")
