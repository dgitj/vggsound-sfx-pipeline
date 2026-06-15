import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from src.download import download_audio
from tqdm import tqdm

class AudioFilter:
    def __init__(self, config):
        self.model_name = config["models"]["stage2"]
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.threshold_speech = config["filtering"]["speech_threshold"]
        self.threshold_music = config["filtering"]["music_threshold"]
        self.keywords_music = config["filtering"]["keywords_music"]
        self.keywords_speech = config["filtering"]["keywords_speech"]
        self.device = config["compute"]["device"]
        self.batch_size = config["compute"]["batch_size"]
        self.model.to(self.device)

    def _is_sfx(self, probs):
        id2label = self.model.config.id2label
        for idx, prob in enumerate(probs):
            label = id2label[idx].lower()
            if any(k in label for k in self.keywords_speech) and prob > self.threshold_speech:
                return False
            if any(k in label for k in self.keywords_music) and prob > self.threshold_music:
                return False
        return True

    def filter(self, sfx_sample):
        downloaded = []
        for _, row in tqdm(sfx_sample.iterrows(), total=len(sfx_sample), desc="Downloading"):
            try:
                audio = download_audio(row["youtube_id"], row["start_seconds"])
                downloaded.append((row, audio))
            except Exception as e:
                print(f"Skipping {row['youtube_id']}: {e}")

        kept_clips = []
        for i in tqdm(range(0, len(downloaded), self.batch_size), desc="Classifying"):
            batch = downloaded[i:i + self.batch_size]
            audios = [audio for _, audio in batch]

            inputs = self.feature_extractor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for j, (row, audio) in enumerate(batch):
                if self._is_sfx(probs[j]):
                    kept_clips.append({
                        "youtube_id": row["youtube_id"],
                        "label": row["label"],
                        "audio": audio
                    })

        return kept_clips