from datasets import load_dataset
import json
import random
from pathlib import Path

LANG = "ta"
TOTAL_SAMPLES = 5500
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


random.seed(SEED)

print("Loading Common Voice Tamil (subset, non-streaming)...")


dataset = load_dataset(
    "fsicoli/common_voice_19_0",
    LANG,
    split=f"train[:{TOTAL_SAMPLES}]"
)

samples = []

for example in dataset:
    if not example.get("sentence"):
        continue

    audio = example.get("audio")
    if not isinstance(audio, dict) or "path" not in audio:
        continue

    samples.append({
        "audio": audio["path"],
        "text": example["sentence"]
    })

print(f"Collected {len(samples)} samples")

random.shuffle(samples)

train_end = int(TRAIN_RATIO * len(samples))
val_end = train_end + int(VAL_RATIO * len(samples))

train = samples[:train_end]
val = samples[train_end:val_end]
test = samples[val_end:]

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_json(train, OUTPUT_DIR / "train.json")
save_json(val, OUTPUT_DIR / "val.json")
save_json(test, OUTPUT_DIR / "test.json")

print("Saved:")
print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")
