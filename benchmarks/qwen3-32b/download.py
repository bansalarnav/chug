"""Download one pinned checkpoint; no second copy of its weights."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root = Path(os.environ['BENCH_ROOT'])
metadata = root / 'results/model.json'
revision = json.loads(metadata.read_text())['revision'] if metadata.exists() else HfApi().model_info('Qwen/Qwen3-32B').sha
metadata.write_text(json.dumps({'model': 'Qwen/Qwen3-32B', 'revision': revision}, indent=2) + '\n')
path = snapshot_download('Qwen/Qwen3-32B', revision=revision, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'], max_workers=2)
(root / 'results/model-path.txt').write_text(path + '\n')
print(path, flush=True)
