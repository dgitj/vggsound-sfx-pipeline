# VGGSound SFX Pipeline

A three-stage filtering pipeline that identifies sound effect clips in the VGGSound dataset and generates a short text description for each one. The output is a `sfx_filtered.jsonl` file with one entry per clip.

---

## The Task

VGGSound contains ~200k YouTube clips across 309 audio categories — a mix of speech, music, and sound effects. The goal is to isolate only the sound effect clips and describe what each one sounds like.

---

## Approach

The pipeline follows a hierarchical filtering strategy: eliminate unwanted clips using the cheapest method first, and only run expensive models on what survives. This keeps compute costs low and makes the pipeline practical at scale.

```
Raw VGGSound CSV
      │
      ▼
┌─────────────────────┐
│ Stage 1: Metadata   │  Keyword matching on labels
│ Filter              │  Drops ~40-50% of clips instantly
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Stage 2: Audio      │  AST model inference
│ Filter              │  Catches speech/music metadata missed
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Stage 3: Captioner  │  CLAP zero-shot matching
│                     │  Creates text descriptions for audio content
└─────────────────────┘
      │
      ▼
sfx_filtered.jsonl
```

---

## Stage 1: Metadata Filter

The pipeline processes all samples in the VGGSound dataset. It reads the VGGSound CSV and discards any clip whose label contains a list of speech or music keywords (e.g. "speech", "talking", "guitar", "singing") defined in the config file. The keyword list can be further extended and tuned to further improve the performance.

---

## Stage 2: Audio Filter

For each clip that passes Stage 1, downloads the audio via `yt-dlp` (links that are currently unavailable on YouTube are logged and skipped gracefully without stoppping the pipeline) and runs it through the Audio Spectrogram Transformer (AST) a model trained on AudioSet's 527 audio classes.

Rather than checking only top-k predictions, the pipeline scans all 527 class probabilities. If any class label matching speech or music keywords exceeds the configured threshold (can be tuned in the config file), the clip is discarded. 


`speech_threshold` and `music_threshold` control the sensitivity of Stage 2. Lower values increase precision (fewer false passes) at the cost of recall (more legitimate SFX rejected). 

---

## Stage 3: Captioner

Generates a text description for each clip that survived Stages 1 and 2, using CLAP (Contrastive Language-Audio Pretraining).

CLAP scores the audio against a vocabulary of ~150 candidate descriptions derived from VGGSound's own SFX labels (music and speech labels excluded). The best-matching candidate becomes the caption.

CLAP was the pragmatic choice for me and it worked well in my evaluations. The trade-off is that captions are constrained to a fixed vocabulary rather than free-form text. If high-quality captions are needed, I would recommend testing more sophisticated generative models.

---

## Hardware and Scalability

The demo runs on a single machine (I used a mac mini) after processing all samples in Stage 1, I sampled 1000 samples randomly to demonstrate the feasibility of the following pipeline. The pipeline is embarrasingly parallel and can scale to the full dataset (or larger). For example, on a SLURM cluster batches can be processed in parallel without synchronization.

Currently, audio downloading in Stage 2 is sequential. Future work should parallelizing downloads e.g., by using a ThreadPoolExecutor.

Stage 1 (keyword matching) and Stage 2 (AST classification) are cheap enough to run on CPU worker pools. To process potentially longer clips in Stage 3 (CLAP captioning) I recommend accelerating model inference by using GPUs.


## Quality Evaluation

I manually verified 20 randomly sampled clips — 5 were unavailable due to dead links, leaving 15 evaluable. Of those, 12 were correctly handled by the pipeline. Three difficult samples were incorrectly classified: For example, 'people whispering' slipped through Stage 2 and 'playing table tennis' was classified as music. 

With further refinements of the keywords and tuning of the thresholds, I believe this can be addressed. Caption quality showed sometimes even better quality than the original labels and sometimes were a bit off. I believe this can be addressed by giving CLAP a larger dictionary or upgrading to a more sophisticated text description model.

A more systematic evaluation would involve building a small labeled held-out set and computing precision/recall for the filtering stages independently, which would also enable threshold tuning in a principled way.

---

## Setup

```bash
# Create environment
conda create -n vggsound-sfx python=3.11
conda activate vggsound-sfx

# Install dependencies
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

---

## Configuration

All pipeline settings live in `config.yaml`:

```yaml
paths:
  input: "vggsound.csv"        # Path to VGGSound CSV
  output: "output/sfx_filtered.jsonl"
  data: "data"                 # Directory for downloaded audio

sampling:
  n_samples: 1000
  random_seed: 42

filtering:
  keywords_music: [...]        # Labels triggering music rejection
  keywords_speech: [...]       # Labels triggering speech rejection
  speech_threshold: 0.5        # AST confidence threshold for speech
  music_threshold: 0.5         # AST confidence threshold for music

models:
  stage2: "MIT/ast-finetuned-audioset-10-10-0.4593"
  stage3: "laion/clap-htsat-unfused"

compute:
  device: "cpu"                # "cuda" for GPU
  batch_size: 32
  num_workers: 4
```

---

## How to Run

**Full pipeline:**
```bash
python run_pipeline.py --config config.yaml
```

---

## Limitations & Future Work

There are several limitations and potential optimization steps on high to low level, that can further extend this pipeline. I would love to discuss them in the interview.
