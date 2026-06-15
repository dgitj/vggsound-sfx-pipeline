import yaml
import argparse
from src.metadata_filter import MetadataFilter
from src.audio_filter import AudioFilter
from src.captioning import Captioner
from src.utils import write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    
    mf = MetadataFilter(config)
    all_sfx_clips = mf.filter(config["paths"]["input"])
    print(f"After Stage 1: {len(all_sfx_clips)} clips")

    n_samples = config["sampling"]["n_samples"]
    n_samples = min(n_samples, len(all_sfx_clips))
    clips = all_sfx_clips.sample(n=n_samples, random_state=config["sampling"]["random_seed"])
    print(f"Sampling {len(clips)} for Stage 2")

    af = AudioFilter(config)
    clips = af.filter(clips)
    print(f"After Stage 2: {len(clips)} clips")

    cp = Captioner(config)
    results = cp.caption(clips)
    print(f"After Stage 3: {len(results)} results")

    write_jsonl(results, config["paths"]["output"])

if __name__ == "__main__":
    main()


