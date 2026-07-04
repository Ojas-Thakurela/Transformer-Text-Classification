# Transformer Showdown: BERT vs RoBERTa vs XLNet on Twitter Emotion Classification

Fine-tunes three transformer architectures — **BERT**, **RoBERTa**, and **XLNet** — on the
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) Twitter dataset (6 emotion
classes: sadness, joy, love, anger, fear, surprise), using **PyTorch** + **HuggingFace `Trainer`**,
optimized to run efficiently on a **free Google Colab GPU (T4)**.

## Colab-specific optimizations used

| Optimization | Where | Why it matters on Colab |
|---|---|---|
| Mixed precision (fp16 / bf16) | `TrainingArguments(fp16=..., bf16=...)` | Roughly halves memory use and speeds up matmuls on Tensor Cores. Auto-detects bf16 on A100 (Colab Pro), falls back to fp16 on T4. |
| Dynamic padding | `DataCollatorWithPadding` | Pads each batch only to its own longest sequence instead of a fixed length — less wasted compute. |
| Length-grouped batching | `group_by_length=True` | Buckets similar-length sequences together, further minimizing padding waste. |
| Gradient accumulation | `gradient_accumulation_steps` | Simulates a larger effective batch size without exceeding a T4's ~15GB memory (used for XLNet, the most memory-hungry of the three). |
| Streamed eval accumulation | `eval_accumulation_steps` | Moves eval logits to CPU incrementally instead of holding them all in GPU memory. |
| TF32 matmuls | `torch.backends.cuda.matmul.allow_tf32` | Free speedup on Ampere+ GPUs (A100); no-op on T4. |
| Explicit memory cleanup | `del trainer, model; gc.collect(); torch.cuda.empty_cache()` after each model | Without this, running BERT → RoBERTa → XLNet back-to-back in one Colab session can exhaust GPU memory. |
| Early stopping | `EarlyStoppingCallback` | Avoids wasting Colab GPU time once validation F1 stops improving. |

## Project structure

```
emotion_transformer_project/
├── emotion_transformers_pytorch.ipynb   # Self-contained Colab notebook (recommended starting point)
├── requirements.txt
├── report_template.md                   # Written report/comparison template to fill in with your results
├── README.md
└── src/
    ├── data_utils.py                    # Dataset loading + tokenization (dynamic padding)
    ├── train_utils.py                   # Trainer-based fine-tune + evaluate routine, Colab-optimized
    ├── compare_utils.py                 # Comparison table + matplotlib visualizations
    └── main.py                          # Script entry point for local/cloud GPU runs
```

## Option A — Run in Google Colab (recommended)

1. Upload `emotion_transformers_pytorch.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime → Change runtime type → T4 GPU** (or an A100 if you have Colab Pro — bf16
   kicks in automatically).
3. Run all cells top to bottom. The first cell installs dependencies.
4. Expect roughly **8–15 minutes per model** for 2 epochs on a T4 with these optimizations —
   noticeably faster and more memory-safe than naive fixed-padding fp32 training.

Outputs (`model_comparison.csv`, chart `.png` files) land in the Colab session's file browser —
download them before the runtime resets, or mount Google Drive.

## Option B — Run locally / on your own GPU machine

```bash
cd emotion_transformer_project
pip install -r requirements.txt
cd src
python main.py
```

Produces the same CSV + PNG outputs as the notebook, in the `src/` directory.

## If you hit a CUDA out-of-memory error

- Lower `per_device_batch_size` and raise `grad_accum_steps` proportionally (keeps the same
  effective batch size with less peak memory), e.g. `per_device_batch_size=8, grad_accum_steps=4`.
- XLNet is the most memory-hungry of the three — the notebook and `main.py` already give it a
  smaller per-device batch (16) with 2x gradient accumulation by default.
- Restart the Colab runtime between attempts — GPU memory from a crashed run isn't always
  freed automatically.

## Models used

| Model   | Checkpoint            | Key architectural idea                                                        |
|---------|-----------------------|----------------------------------------------------------------------------|
| BERT    | `bert-base-uncased`   | Bidirectional encoder, pretrained via masked-LM + next-sentence-prediction |
| RoBERTa | `roberta-base`        | BERT variant: no NSP, more data, larger batches, dynamic masking          |
| XLNet   | `xlnet-base-cased`    | Permutation-based autoregressive pretraining, captures bidirectional context without masking |

## After running

Fill in `report_template.md` with the numbers and charts your run produced.
