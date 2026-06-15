import pandas as pd

class MetadataFilter:
    def __init__(self, config):
        self.keywords_music = config["filtering"]["keywords_music"]
        self.keywords_speech = config["filtering"]["keywords_speech"]
    
    def filter(self, csv_path):
        df = pd.read_csv(csv_path, names=["youtube_id", "start_seconds", "label", "split"])

        df["category"] = df["label"].apply(
            lambda label: "speech" if any(k in label.lower() for k in self.keywords_speech)
            else "music" if any(k in label.lower() for k in self.keywords_music)
            else "sfx"
    )
        return df[df["category"] == "sfx"]