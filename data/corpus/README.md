# Speaker Role Detection and Segmented Analysis in Customer Service Calls

**FYP2 — Muhammad Eiqbal Bin Hasbollah (2216911)**  
Department of Mechatronics Engineering, IIUM · 2026  
Supervisor: Prof. Dr. Ir. Siti Fauziah Bt. Toha @ Tohara

---

## What This System Does

An automated AI pipeline that takes raw customer service call recordings and:

1. Transcribes speech using OpenAI Whisper (large-v3-turbo model)
2. Separates speakers using pyannote.audio neural diarization
3. Classifies each speaker as **Agent** or **Customer** using three methods
4. Detects rude behaviour, call outcome, and generates call summaries
5. Computes QA analytics — talk ratio, sentiment, compliance, QA score
6. Validates accuracy against human-verified transcripts
7. Displays everything on an interactive Streamlit dashboard

---

## System Architecture

```
Audio (.wav/.mp3)
      │
      ▼
Phase 1 — Preprocessing + Transcription
  Resample → 16 kHz mono
  Spectral gating noise reduction
  Whisper large-v3-turbo (Google Colab GPU) → transcription
  pyannote.audio 3.1 → speaker diarization
  fix_speakers.py → re-cluster via d-vector if needed
      │
      ▼
Phase 2 — Speaker Role Detection (3 Methods)
  Method 1 (Keyword-Lexical)   → keyword density classifier
  Method 2 (Acoustic DNN)      → MLP on MFCC + d-vector features
  Method 3 (Adaptive LLM)      → Llama 3.3 70B via Groq API
                                   (adaptive Two-Pass + Per-Segment)
      │
      ▼
Phase 3 — Analytics
  Talk-time ratio, silence %, turn-taking
  VADER sentiment trajectory
  SOP compliance checklist + risk flags
  Rude behaviour detection (Agent + Customer, severity levels)
  Call outcome detection (Resolved / Unresolved / Escalated / Transferred)
  Call summary generation (LLM)
  Composite QA score with rudeness auto-fail penalty
      │
      ▼
Phase 4 — Evaluation
  Accuracy / Precision / Recall / F1
  Paired t-test (Method 3 vs Method 1)
  Pearson r (system QA vs human QA)
  RTF (Real-Time Factor)
      │
      ▼
Phase 5 — Dashboard
  streamlit run dashboard/app.py
```

---

## Current Results (30 Calls)

```
Method                        Accuracy  Precision  Recall     F1
──────────────────────────────────────────────────────────────────
Method 1 (Keyword-Lexical)      71.3%     66.2%    87.3%   73.9%
Method 2 (Acoustic DNN)         72.6%     70.2%    88.7%   76.3%
Method 3 (LLM — Llama 3.3 70B) 93.1%     95.3%    92.9%   94.0%

Paired t-test (Method 3 vs Method 1):
  t = 14.376,  p < 0.001  → Method 3 SIGNIFICANTLY better (α = 0.05)

System vs Human QA Correlation:
  Pearson r = 0.35  (p = 0.057)  — weak positive correlation

RTF: 0.013 (real-time capable — 77.5× faster than audio duration)
Total audio: 4,710.74 seconds (78.5 minutes) across 30 calls
Local processing time: 60.82 seconds (excludes Colab transcription)
```

Per-call accuracy (Method 3 — segment-level label comparison):

| Call ID | M3 Accuracy | M3 F1 | Language | Type |
|---|---|---|---|---|
| eng_prof_01 | 100.0% | 100.0% | English | Professional inbound |
| eng_prof_02 | 100.0% | 100.0% | English | Professional inbound |
| eng_prof_03 | 100.0% | 100.0% | English | Professional inbound |
| eng_prof_04 | 100.0% | 100.0% | English | Professional inbound |
| eng_prof_05 | 91.3% | 90.0% | English | Professional inbound |
| eng_rudeagt_01 | 81.2% | 82.4% | English | Rude agent |
| eng_rudeagt_02 | 97.1% | 97.3% | English | Rude agent |
| eng_rudeagt_03 | 86.8% | 88.9% | English | Rude agent |
| eng_rudeagt_04 | 100.0% | 100.0% | English | Rude agent |
| eng_rudecust_01 | 97.8% | 98.1% | English | Rude customer |
| eng_rudecust_02 | 92.3% | 94.1% | English | Rude customer |
| eng_rudecust_03 | 100.0% | 100.0% | English | Rude customer |
| eng_rudecust_04 | 88.6% | 88.2% | English | Rude customer |
| eng_rudecust_05 | 96.7% | 97.0% | English | Rude customer |
| long_01 | 87.7% | 90.7% | English | Long call |
| manglish_01 | 79.2% | 80.5% | Manglish | Code-switching |
| manglish_02 | 93.5% | 93.3% | Manglish | Code-switching |
| manglish_03 | 98.7% | 98.8% | Manglish | Code-switching |
| manglish_04 | 95.8% | 96.0% | Manglish | Code-switching |
| manglish_05 | 92.7% | 93.3% | Manglish | Code-switching |
| my_prof_01 | 100.0% | 100.0% | Malay | Professional inbound |
| my_prof_02 | 100.0% | 100.0% | Malay | Professional inbound |
| my_prof_03 | 100.0% | 100.0% | Malay | Professional inbound |
| my_prof_04 | 96.7% | 97.0% | Malay | Professional inbound |
| my_prof_05 | 98.1% | 98.4% | Malay | Professional inbound |
| my_rude_01 | 83.6% | 85.3% | Malay | Rude agent |
| my_rude_02 | 70.0% | 77.4% | Malay | Rude agent |
| my_rude_03 | 100.0% | 100.0% | Malay | Rude agent |
| my_sales_01 | 66.1% | 72.5% | Malay | Sales outbound |
| my_sales_02 | 100.0% | 100.0% | Malay | Sales outbound |

**Overall: 93.1% accuracy, 94.0% F1 (30 calls)**

> Hardest cases: `my_rude_02`, `my_sales_01`, `manglish_01` — informal/rude agent speech is
> linguistically indistinguishable from customer speech, defeating lexical and acoustic signals.

---

## Project Structure

```
fyp2/
│
├── config.py                    ← ALL settings live here
├── main.py                      ← Run this to process all calls
├── train.py                     ← Train the Method 2 acoustic DNN
├── evaluate.py                  ← Evaluate accuracy vs human_validation_study.csv
├── compare_labels.py            ← Segment-by-segment label diff
├── combine_results.py           ← Merge day1.json + day2.json
├── qa_scorer.py                 ← Industry-standard QA scoring rubric
├── fix_speakers.py              ← Re-cluster diarization via d-vectors
├── preflight_check.py           ← Validate files/keys before running
├── run_tracker.py               ← Track call processing status
├── run_day.bat                  ← Automated batch runner (Windows)
├── gpu_check.py                 ← System readiness check
├── requirements.txt
└── README.md
│
├── data/                        ← Drop your .wav / .mp3 files here
│
├── colab_transcripts/           ← Diarized JSONs from Colab (never auto-deleted)
│   └── <call_id>_diarized.json
│
├── human_transcripts/           ← Human-verified per-segment labels (one CSV per call)
│   └── <call_id>.csv
│
├── human_validation_study.csv   ← Ground truth labels + human QA scores (30 calls)
│
├── keywords/
│   ├── agent_keywords.json      ← Agent lexicon (English + Malay + Manglish)
│   └── customer_keywords.json   ← Customer lexicon
│
├── preprocessing/
│   ├── audio_processor.py       ← Noise reduction, normalisation, SNR
│   ├── transcriber.py           ← Whisper ASR + pyannote diarization
│   └── malay_corrections.py     ← Post-transcription Malay text corrections
│
├── methods/
│   ├── method1_lexical.py       ← Keyword density classifier (baseline)
│   ├── method2_acoustic.py      ← PyTorch DNN on MFCC + d-vector features
│   └── method3_llm.py           ← Adaptive LLM classifier (Llama 3.3 70B via Groq)
│
├── analytics/
│   ├── talk_ratio.py            ← Talk-time ratio, QA score
│   ├── sentiment.py             ← VADER sentiment trajectory
│   ├── compliance.py            ← SOP checklist, risk flagging, call outcome
│   ├── rude_behavior.py         ← Rude behaviour detection with severity levels
│   └── advanced.py              ← Response time, interruption detection, WER
│
├── evaluation/
│   ├── metrics.py               ← Accuracy, F1, t-test, Pearson r, RTF
│   └── validator.py             ← Text-similarity matching vs ground truth
│
├── dashboard/
│   └── app.py                   ← Streamlit QA dashboard
│
├── models/
│   ├── acoustic_model.pth       ← Trained DNN weights
│   └── scaler.npy               ← StandardScaler parameters
│
└── outputs/
    ├── latest/                  ← Default output folder (overwritten each run)
    ├── calls/                   ← Per-call output subfolders
    │   └── <call_id>/
    │       ├── *_waveform.png
    │       ├── *_spectrogram.png
    │       ├── *_sentiment.png
    │       ├── *_m1_confidence.png
    │       ├── *_m1_keywords.png
    │       ├── *_method_comparison.png
    │       └── *_transcript.txt
    └── run_01/                  ← Named archived runs (set RUN_NAME in config.py)
```

---

## Setup

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA (recommended; Google Colab T4 GPU used for Whisper transcription)
- Conda environment `fyp2`
- ffmpeg on PATH
- Groq API key (free at https://console.groq.com)
- HuggingFace token (free at https://huggingface.co/settings/tokens)

### Install packages
```bash
conda activate fyp2
pip install -r requirements.txt
```

### Configure API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
HUGGINGFACE_TOKEN=hf_your_token_here
```

---

## Running the System

### Hybrid workflow (Colab for transcription, local for everything else)

**Step 1 — Transcribe on Colab** (once per audio file, requires GPU)
1. Upload `.wav` files to Google Drive
2. Run `FYP_Transcribe_Colab.ipynb` on Colab (T4 GPU)
3. Download `*_diarized.json` files → put in `colab_transcripts/`

**Step 2 — Pre-flight check**
```bash
python preflight_check.py --day 1
```

**Step 3 — Run pipeline (Day 1: first 15 calls)**
```bash
run_day.bat 1
```

**Step 4 — Run pipeline (Day 2: remaining 15 calls)**
```bash
run_day.bat 2
```
After Day 2, `combine_results.py`, `compare_labels.py`, `evaluate.py`, and `qa_scorer.py` run automatically.

**Step 5 — Dashboard**
```bash
streamlit run dashboard/app.py
```

### Manual run (single call)
```bash
python main.py --skip_transcription --call_id eng_prof_01
```

### Useful flags
```bash
python main.py --skip_transcription              # use existing Colab JSONs (normal mode)
python main.py --skip_acoustic                   # skip Method 2 DNN (faster)
python main.py --whisper_model large-v3-turbo    # override Whisper model size
python main.py --call_id my_rude_01              # process one call only
python run_tracker.py                            # check processing status
python run_tracker.py --reset                    # reset all to pending
```

---

## Training Method 2 (Acoustic DNN)

Run once before the first pipeline run. Uses `human_transcripts/*.csv` with actual audio timestamps.

```bash
python train.py
python train.py --epochs 100
python train.py --no_augment    # faster, less accurate
```

Augmentation applied: Gaussian noise, time-stretch ×0.9/×1.1, pitch shift ±2 semitones.
Each original sample → 5 augmented copies (6× dataset expansion).

---

## Adding a New Call

1. Put `.wav` file in `data/`
2. Transcribe on Colab → put `<call_id>_diarized.json` in `colab_transcripts/`
3. Create `human_transcripts/<call_id>.csv`:
```csv
segment_id,role,text,start,end
1,Agent,Thank you for calling,0.0,3.2
2,Customer,I need help,3.5,5.1
```
4. Add rows to `human_validation_study.csv`
5. Run pipeline

---

## QA Scoring Rubric (Industry Standard)

Based on Balto.ai (2025), Globalify (2026), Calabrio (2026):

| Component | Weight | Description |
|---|---|---|
| Resolution | 30% | Resolved=100, Transferred=70, Escalated=50, Unresolved=0 |
| Compliance | 25% | SOP checklist (greeting, closing, disclaimer, identity) |
| Sentiment | 20% | Agent sentiment weighted 60%, customer 40% |
| Communication | 15% | Talk ratio balance + turn-taking + response time |
| Professionalism | 10% | Agent rudeness level |

**Auto-fail rule:** Agent HIGH rudeness → score capped at 20/100.

---

## Analytics Features

| Feature | Description |
|---|---|
| QA Score | Composite 0–100 with rudeness auto-fail penalty |
| Rude Behaviour | NONE / LOW / MEDIUM / HIGH for Agent and Customer |
| Call Outcome | Resolved / Unresolved / Escalated / Transferred |
| Call Summary | 3-line LLM summary (Topic / Summary / Outcome) |
| Sentiment | VADER compound trajectory per speaker |
| Compliance | SOP checklist with partial scoring |
| Talk Ratio | Agent / Customer / Silence % breakdown |
| WER | Word Error Rate vs human transcript (per call) |
| Response Time | Agent response latency after customer turn |

---

## Token Usage (Groq Free Tier)

- Daily limit: 100,000 tokens
- Per call: ~3,000–5,000 tokens
- 30 calls total: split into 2 days of 15 calls each (~50,000 tokens/day)
- Quota resets at 00:00 UTC (08:00 Malaysia time)

---

## Common Errors

| Error | Fix |
|---|---|
| `No module named dotenv` | `conda activate fyp2` first |
| `Rate limit 429` | Wait for Groq reset at 08:00 Malaysia time |
| `CUDA out of memory` | Use `python main.py --skip_acoustic` |
| `UnicodeDecodeError` | Ensure files use UTF-8 encoding |
| `No audio files found` | Check `.wav` files are in `data/` |
| Dashboard shows no data | Run pipeline first, then refresh |
| Single-speaker diarization | Run `python fix_speakers.py --call_id <id>` |
| Manglish/Malay misclassified | Known limitation — informal agent speech defeats M1/M2 |
