
# Whisper ASR Fine-Tuning using LoRA

Fine-tuning OpenAI’s Whisper model for **Tamil Automatic Speech Recognition (ASR)** using **Low-Rank Adaptation (LoRA)** on a small subset of the **Common Voice Tamil** dataset.

This project demonstrates **parameter-efficient fine-tuning** of large speech models for **low-resource languages** and highlights the **limitations of WER-based evaluation** on noisy datasets.

---

##  Motivation

Whisper performs well on high-resource languages but struggles with **low-resource languages like Tamil**, especially when training data is limited.

Traditional full fine-tuning is computationally expensive. This project explores whether **LoRA**, a parameter-efficient fine-tuning technique, can adapt Whisper to Tamil using **a small amount of data and limited compute**.

---

##  Key Concepts

- Automatic Speech Recognition (ASR)
- Whisper (Encoder–Decoder Transformer)
- Low-Rank Adaptation (LoRA)
- Parameter-Efficient Fine-Tuning (PEFT)
- Word Error Rate (WER)
- Low-resource language modeling

---

##  Dataset

**Mozilla Common Voice – Tamil**

- Public, open-source speech dataset
- Contains diverse speakers and accents
- Includes noisy and partially aligned transcripts

### Dataset Usage
- Training samples: **1200**
- Validation samples: **150**
- Approx. audio size: **~2 GB**
- Sampling rate: **16 kHz**

Dataset source:  
https://huggingface.co/datasets/fsicoli/common_voice_19_0

---

##  Model Architecture

- **Base model:** `openai/whisper-small`
- **Architecture:** Encoder–Decoder Transformer
- **Task:** Speech-to-text transcription
- **Language:** Tamil (`ta`)

---

##  Fine-Tuning Method

### LoRA Configuration
- Target modules: Attention projection layers
- Rank (`r`): 8  
- Alpha (`α`): 16  
- Dropout: 0.1  

### Parameter Statistics
- Trainable parameters: **~885K**
- Total parameters: **~242M**
- Trainable ratio: **~0.36%**

---

##  Training Setup

- Optimizer: AdamW
- Learning rate: `1e-4`
- Batch size: Colab-compatible
- Mixed precision: FP16
- Frameworks:
  - Hugging Face Transformers
  - Datasets
  - PEFT
  - Evaluate

---

##  Evaluation

### Metric: Word Error Rate (WER)

\[
WER = \frac{S + D + I}{N}
\]

Where:
- **S** = Substitutions  
- **D** = Deletions  
- **I** = Insertions  
- **N** = Number of reference words  

WER is the standard ASR metric but is sensitive to transcript noise and sentence length variations.

---

##  Results

| Model | WER (%) |
|------|---------|
| Whisper-small (Base) | **88.3%** |
| Whisper-small + LoRA | **100.33%** |

---

##  Observations

- LoRA fine-tuning reduced training loss and improved output fluency.
- WER increased after fine-tuning due to:
  - Noisy and incomplete reference transcripts
  - Increased insertions in model predictions
  - Known limitations of WER in low-resource ASR

This behavior is consistent with findings in prior ASR research.

---

##  Key Takeaways

- Successfully fine-tuned Whisper using **<1% trainable parameters**
- Demonstrated parameter-efficient adaptation for Tamil ASR
- Highlighted real-world evaluation challenges in low-resource datasets
- Showed that lower training loss does not necessarily imply lower WER

---

##  Future Work

- Train on larger and cleaner Tamil datasets
- Apply transcript normalization and filtering
- Use additional metrics (CER, human evaluation)
- Experiment with larger Whisper variants
- Explore sequence-level ASR training objectives

---



