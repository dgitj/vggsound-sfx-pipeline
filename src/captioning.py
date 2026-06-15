import torch
import torchaudio
from transformers import ClapModel, ClapProcessor
import pandas as pd

class Captioner:
    def __init__(self, config):
        self.model_name = config["models"]["stage3"]
        self.model = ClapModel.from_pretrained(self.model_name)
        self.processor = ClapProcessor.from_pretrained(self.model_name)
        self.keywords_speech = config["filtering"]["keywords_speech"]
        self.keywords_music = config["filtering"]["keywords_music"]
        self.device = config["compute"]["device"]
        self.model.to(self.device)
        self.candidates = self._build_candidates(config["paths"]["input"])

    def _build_candidates(self, csv_path):
        df = pd.read_csv(csv_path, names=["youtube_id", "start_seconds", "label", "split"])
        all_labels = df["label"].unique().tolist()
        sfx_labels = [
            label for label in all_labels
            if not any(k in label.lower() for k in self.keywords_speech)
            and not any(k in label.lower() for k in self.keywords_music)
        ]
        return [f"The sound of {label}" for label in sfx_labels]

    def caption(self, kept_clips):
        results = []

        for clip in kept_clips:
            audio = clip["audio"]
            audio_48k = torchaudio.functional.resample(torch.tensor(audio), orig_freq=16000, new_freq=48000).numpy()

            inputs = self.processor(
                audio=audio_48k,
                text=self.candidates,
                return_tensors="pt",
                sampling_rate=48000,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_audio[0]
                probs = torch.softmax(logits, dim=-1)

            best_idx = probs.argmax().item()
            description = self.candidates[best_idx]

            results.append({
                "video_id": clip["youtube_id"],
                "audio_text_description": description
            })

        return results