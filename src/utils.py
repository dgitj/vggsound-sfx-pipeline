import json
import os

def write_jsonl(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in results:
            line= json.dumps(item)
            f.write(line + "\n")