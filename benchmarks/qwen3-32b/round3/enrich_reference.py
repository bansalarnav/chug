"""Additional probability-aware comparisons for saved cached/full-prefix logits."""
import json,os
from pathlib import Path
import numpy as np
root=Path(os.environ['BENCH_ROOT'])/'round3/fixtures';meta=json.loads((root/'reference.json').read_text())
for item in meta['fixtures']:
 arrays=np.load(root/item['file'])
 for check in item['checks']:
  step=check['step'];full=arrays[f'continuation_logits_{step}'].astype(np.float64);cached=arrays[f'cached_logits_{step}'].astype(np.float64)
  d=cached-full;shift=float(d.mean());centered=float(np.sqrt(np.mean((d-shift)**2)))
  def logsoft(x):
   z=x-x.max();return z-np.log(np.exp(z).sum())
  logp,logq=logsoft(full),logsoft(cached);p=np.exp(logp)
  check.update(mean_logit_shift=shift,centered_rmse=centered,
    centered_relative_rmse=centered/max(float(full.std()),1e-12),
    kl_full_to_cached_nats=float(np.sum(p*(logp-logq))),
    top10_overlap=len(set(np.argpartition(full,-10)[-10:])&set(np.argpartition(cached,-10)[-10:]))/10)
meta['comparison_note']='Uncentered logit error includes common offsets, which softmax ignores. Centered error, KL divergence and top-10 overlap are also provided; no universal pass tolerance is asserted.'
(root/'reference.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
print('Enriched',len(meta['fixtures']),'saved fixtures without rerunning the model.')
