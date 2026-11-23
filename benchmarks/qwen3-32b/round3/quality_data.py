#!/usr/bin/env python3
"""Pin a small public validation slice for relative quantization perplexity."""
import hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
import pyarrow.parquet as pq
root=Path(os.environ['BENCH_ROOT'])/'round3';repo='Salesforce/wikitext'
saved=root/'fixtures/quality-data.json'
revision=json.loads(saved.read_text())['revision'] if saved.exists() else HfApi().dataset_info(repo).sha
name='wikitext-2-raw-v1/validation-00000-of-00001.parquet'
file=hf_hub_download(repo_id=repo,repo_type='dataset',filename=name,revision=revision)
texts=pq.read_table(file,columns=['text']).column('text').to_pylist()
selected=[];characters=0
for text in texts:
 if text.strip():
  selected.append(text);characters+=len(text)
 if characters>=30000:break
path=root/'fixtures/wikitext-validation.txt';path.write_text('\n'.join(selected))
(root/'fixtures/quality-data.json').write_text(json.dumps(dict(dataset=repo,revision=revision,split='validation',source_file=name,
 rows=len(selected),characters=characters,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 purpose='Small fixed public validation slice for relative BF16/Q8/Q4 perplexity; not a broad model-quality evaluation.'),indent=2)+'\n')
print(path,characters,flush=True)
